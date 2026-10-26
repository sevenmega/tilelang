"""Standalone test for the fused qkv-projection / qk-norm / RoPE / cache-write kernel.

Compiles the **real tilelang DSL kernel** ``fused_qkv_rope`` (see
:mod:`tilelang.tpu.kernels.fused_qkv_rope`) through the generic TIR->PPL
translator and checks it against a plain torch reference that mirrors what
Qwen3's pre-attention chain does today (bf16 qkv projection, fp32 RMSNorm per
head, NeoX RoPE, scatter into the NHD cache).

    X       [Tp, H]           normalized hidden state
    W       [Ntot, H]         qkv_proj.weight, rows [q | k | v], head-major
    CosRows [Tp, D]           fp32: [:, :64] cos, [:, 64:] sin, per token
    Gamma   [2, D]            fp32 q_norm / k_norm weights
    KVRow   [2*Hkv, Tp]       uint32 flat cache rows: slot*2*Hkv + 2*h + half
    Kc/Vc   [Slots*2*Hkv, 64] the cache's flat 64-wide row view, written in place
    QOut    [Tp, Hq*D]        packed queries for the attention kernel

Three things are asserted, and each fails differently:

* ``Q``, the scattered ``K`` and the scattered ``V`` match the reference -- this
  is the arithmetic (projection tiles, fp32 reduction, half-split RoPE).
* Flat rows the index table never names are untouched, bit for bit -- this is
  the scatter's row addressing (a wrong destination stride or base writes
  garbage somewhere else while still looking right at the addressed rows).
* With ``T`` not a multiple of ``block_M``, the padding rows write only to the
  reserved pad slot -- the contract the driver relies on to over-read ``X``.

The arithmetic is checked twice.  The kernel normalizes its **fp32** projection
accumulator; the model normalizes a **bf16-rounded** one, so the two differ by
that single rounding.  The fp32 comparison is the assertion (it is what the
kernel should compute); the model-path comparison is printed as the number that
predicts whether greedy decoding changes.

Run on real hardware under the PPL env (see the ``ppl-tpu-run-recipe`` memo):

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_qkv_rope.py --run

Without ``--run`` it only emits + prints the kernel source (compile smoke test).
"""

import os
import sys

import torch

from tilelang.tpu.kernels.fused_qkv_rope import fused_qkv_rope

H, HQ, HKV, D = 1024, 16, 8, 128
HALF = D // 2
NTOT = (HQ + 2 * HKV) * D  # q | k | v rows of the fused qkv weight
EPS = 1e-6
BLOCK_M = 64
SLOTS_TOTAL = 256


def _logical_device() -> int:
    """The torch *logical* device id to run on (see ``test_flash_attention_paged``)."""
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


def _cos_sin_rows(positions):
    """[len(pos), D] fp32: cos in the first half, sin in the second (NeoX pairs)."""
    inv_freq = 10000.0 ** (-torch.arange(0, HALF, dtype=torch.float32) / HALF)
    ang = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)


def _ref(x, w, gamma, cos_rows, exact_proj):
    """(q [T, HQ*D], k [T, HKV, D], v [T, HKV, D]) as the model produces them.

    ``exact_proj`` = True normalizes the fp32 projection, False the bf16-rounded
    one (what the model actually feeds the norm).
    """
    t = x.shape[0]
    qkv = x.float() @ w.float().t()
    if not exact_proj:
        qkv = qkv.to(torch.bfloat16).float()
    q = qkv[:, : HQ * D].reshape(t, HQ, D)
    k = qkv[:, HQ * D : (HQ + HKV) * D].reshape(t, HKV, D)
    v = qkv[:, (HQ + HKV) * D :].reshape(t, HKV, D)

    def rms(tn, g):
        return tn * torch.rsqrt(tn.pow(2).mean(-1, keepdim=True) + EPS) * g

    def rope(tn):
        lo, hi = tn[..., :HALF], tn[..., HALF:]
        c = cos_rows[:, :HALF].unsqueeze(1)
        s = cos_rows[:, HALF:].unsqueeze(1)
        return torch.cat([lo * c - hi * s, hi * c + lo * s], dim=-1)

    return rope(rms(q, gamma[0])).reshape(t, HQ * D), rope(rms(k, gamma[1])), v


def _cmp(name, got, ref, rel_tol=2e-3, assert_ok=True):
    diff = (got.float() - ref.float()).abs()
    rel = (diff.norm() / ref.float().norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    ok = cos > 0.999 and rel < rel_tol
    verdict = ("ok" if ok else "FAIL") if assert_ok else f"rel<{rel_tol:g}: {ok}"
    print(f"  {name:<30}{verdict:<14}rel={rel:.2e} cos={cos:.6f} "
          f"maxabs={diff.max().item():.4f}")
    return ok


def _run(kernel, t_tok, do_source):
    dev = f"tpu:{_logical_device()}"
    torch.manual_seed(0)
    tp = -(-t_tok // BLOCK_M) * BLOCK_M
    s2 = SLOTS_TOTAL * 2 * HKV
    pad_slot = SLOTS_TOTAL - 1  # the pool's reserved row, excluded from checks

    x = torch.randn(t_tok, H).to(torch.bfloat16)
    w = (torch.randn(NTOT, H) / H**0.5).to(torch.bfloat16)
    gamma = (torch.randn(2, D) * 0.5 + 1.0).float()
    positions = torch.randint(0, 4096, (t_tok,))
    cos_rows = _cos_sin_rows(positions)

    slots = torch.randperm(SLOTS_TOTAL - 1)[:t_tok].int()
    # flat cache row of head h's `half` of token t: slot*2*HKV + 2h + half
    flats = slots.long().unsqueeze(1) * (2 * HKV) + torch.arange(2 * HKV).long()
    kv_row = torch.empty(2 * HKV, tp, dtype=torch.uint32)
    kv_row[:, :t_tok] = flats.t()
    kv_row[:, t_tok:] = pad_slot * (2 * HKV) + torch.arange(2 * HKV).view(-1, 1)

    # Padded token rows read zeros: their heads normalize to 0 (var = eps), their
    # cos row is 0, so no NaN reaches the reserved slot they write to.
    x_src = torch.zeros(tp, H).to(torch.bfloat16)
    x_src[:t_tok] = x
    cos_src = torch.zeros(tp, D)
    cos_src[:t_tok] = cos_rows

    kc0 = torch.randn(s2, HALF).to(torch.bfloat16)
    vc0 = torch.randn(s2, HALF).to(torch.bfloat16)
    kc_dev, vc_dev = kc0.to(dev), vc0.to(dev)

    # ``QOut`` is the PrimFunc's last param, so it is the only output and comes
    # back allocated; the caches are written in place through their pointers.
    q_dev = kernel(
        x_src.to(dev),
        w.to(dev),
        cos_src.to(dev),
        gamma.to(dev),
        kv_row.to(dev),
        kc_dev,
        vc_dev,
    )
    q_out, kc_out, vc_out = q_dev.float().cpu(), kc_dev.float().cpu(), vc_dev.float().cpu()
    if do_source:
        print(kernel.get_kernel_source())

    # Reassemble the two scattered halves into [T, HKV, D].
    def _halves(buf):
        return torch.stack(
            [torch.cat([buf[flats[:, 2 * h]], buf[flats[:, 2 * h + 1]]], dim=-1)
             for h in range(HKV)], dim=1,
        )

    k_got, v_got = _halves(kc_out), _halves(vc_out)

    ok = True
    q_ref, k_ref, v_ref = _ref(x, w, gamma, cos_rows, exact_proj=True)
    print("  -- vs fp32 projection (what the kernel computes)")
    ok &= _cmp("q", q_out[:t_tok], q_ref)
    ok &= _cmp("k (scattered rows)", k_got, k_ref)
    ok &= _cmp("v (scattered rows)", v_got, v_ref)

    q_m, k_m, v_m = _ref(x, w, gamma, cos_rows, exact_proj=False)
    print("  -- vs the model's bf16-rounded projection (not asserted; this is the")
    print("     one-rounding difference the fused path introduces)")
    for name, got, ref in (("q", q_out[:t_tok], q_m), ("k", k_got, k_m), ("v", v_got, v_m)):
        _cmp(name, got, ref, rel_tol=1e-2, assert_ok=False)

    # Only the token slots' rows may change -- plus, when T is not a tile
    # multiple, the reserved pad slot the padding rows are pointed at.
    pad_flats = pad_slot * (2 * HKV) + torch.arange(2 * HKV)
    mask = torch.ones(s2, dtype=torch.bool)
    mask[torch.cat([flats.flatten(), pad_flats])] = False
    untouched = torch.equal(kc_out[mask], kc0.float()[mask]) and torch.equal(
        vc_out[mask], vc0.float()[mask]
    )
    print(f"  {'untouched rows':<30}{'ok' if untouched else 'FAIL':<14}"
          f"({int(mask.sum())} rows)")
    ok &= untouched

    # Padding rows must land on the reserved pad slot and nowhere else.  When T
    # fills the last tile there are none, so the slot stays pristine.
    pad_written = not torch.equal(kc_out[pad_flats], kc0.float()[pad_flats])
    has_pad = t_tok % BLOCK_M != 0
    pad_ok = pad_written == has_pad
    print(f"  {'pad slot written (expect %s)' % has_pad:<30}"
          f"{'ok' if pad_ok else 'FAIL':<14}slot={pad_slot}")
    return ok and pad_ok


def main():
    do_run = "--run" in sys.argv
    do_source = "--source" in sys.argv

    # Two launches: T on a tile boundary and T one tile short of one, which is
    # what exercises the reserved-slot padding path.
    kernels = {t: fused_qkv_rope(block_M=BLOCK_M, dtype="bfloat16") for t in (128, 100)}
    if not do_run:
        for t, k in kernels.items():
            print(f"T={t}: compiled")
            if do_source:
                print(k.get_kernel_source())
        return 0

    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")

    ok = True
    for t_tok, kernel in kernels.items():
        print(f"== T={t_tok}  (padded to {-(-t_tok // BLOCK_M) * BLOCK_M} rows)")
        ok &= _run(kernel, t_tok, do_source)
    print("PASS: fused qkv/rope matches reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
