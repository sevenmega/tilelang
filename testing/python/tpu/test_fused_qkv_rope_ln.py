"""Standalone test for the fused input-norm / qkv / qk-norm / RoPE / cache kernel.

Compiles the **real tilelang DSL kernel** ``fused_qkv_rope_ln`` (see
:mod:`tilelang.tpu.kernels.fused_qkv_rope_ln`) through the generic TIR->PPL
translator and checks it against a plain torch reference of the whole Qwen3
pre-attention chain: residual add already applied by the caller, then
``input_layernorm`` -> ``qkv_proj`` -> per-head RMSNorm -> NeoX RoPE -> scatter
into the NHD paged cache.

    X       [Tp, H]           *un-normalized* hidden (= hidden + residual)
    W       [Ntot, H]         qkv_proj.weight * input_layernorm.weight, bf16
                              (folded once on the host, see the kernel docstring)
    CosRows [Tp, D]           fp32: [:, :64] cos, [:, 64:] sin, per token
    Gamma   [2, D]            fp32 q_norm / k_norm weights
    KVRow   [2*Hkv, Tp]       uint32 flat cache rows: slot*2*Hkv + 2*h + half
    Kc/Vc   [Slots*2*Hkv, 64] the cache's flat 64-wide row view, written in place
    QOut    [Tp, Hq*D]        packed queries for the attention kernel

Two references, because the kernel makes two documented choices:

* **kernel-exact** -- ``q = rstd_in * (x . (gin*W)^T)`` with ``gin*W`` rounded to
  bf16 and handed to the kernel as its weight.  This is what the kernel
  computes: the input ``gamma`` indexes the
  contraction axis so it folds into the weight exactly, and the per-row
  ``rstd`` is applied to the fp32 accumulator rather than to a bf16-rounded
  normalized hidden.  **This is the assertion.**
* **model** -- what ``RMSNorm.forward_native`` + ``F.linear`` do: normalize,
  round to bf16, then project with the unfolded weight, and normalize the
  bf16-rounded projection for qk-norm.  Printed, not asserted: the gap is the
  rounding the fused path trades for one launch.

The scatter invariants from the projection-only test are re-checked: rows the
index table never names stay bit-identical, and the padding rows of a short
batch write only to the pool's reserved pad slot.

Run on real hardware under the PPL env (see the ``ppl-tpu-run-recipe`` memo):

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_qkv_rope_ln.py --run

Without ``--run`` it only compiles (smoke test for the emitter path).
"""

import os
import sys

import torch

from tilelang.tpu.kernels.fused_qkv_rope_ln import fused_qkv_rope_ln

H, HQ, HKV, D = 1024, 16, 8, 128
HALF = D // 2
NTOT = (HQ + 2 * HKV) * D  # q | k | v rows of the fused qkv weight
EPS = 1e-6
LN_EPS = 1e-6
BLOCK_M = 64
SLOTS_TOTAL = 256
REPS = 10


def _logical_device() -> int:
    """The torch *logical* device id to run on (see ``test_flash_attention_paged``)."""
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


def _cos_sin_rows(positions):
    """[len(pos), D] fp32: cos in the first half, sin in the second (NeoX pairs)."""
    inv_freq = 10000.0 ** (-torch.arange(0, HALF, dtype=torch.float32) / HALF)
    ang = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)
    return torch.cat([torch.cos(ang), torch.sin(ang)], dim=1)


def _qk_rope(qkv, gamma, cos_rows, exact_proj):
    """Split the projection and apply qk-RMSNorm + NeoX RoPE to q and k."""
    t = qkv.shape[0]
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


def _ref_kernel_exact(x, w_folded, gamma, cos_rows):
    """What the kernel computes: pre-folded W, rstd applied post-GEMM.

    Only ``v`` carries ``rstd``: it cancels in q/k's own RMSNorm, so the kernel
    never scales them.
    """
    xf = x.float()
    rstd_in = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + LN_EPS)
    qkv = xf @ w_folded.float().t()
    qkv[:, (HQ + HKV) * D :] = qkv[:, (HQ + HKV) * D :] * rstd_in
    return _qk_rope(qkv, gamma, cos_rows, exact_proj=True)


def _ref_model(x, w, gin, gamma, cos_rows):
    """What ``input_layernorm`` + ``qkv_proj`` + ``apply_qk_norm`` + RoPE do."""
    xf = x.float()
    xnorm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + LN_EPS) * gin.float()
    qkv = xnorm.to(torch.bfloat16).float() @ w.float().t()
    return _qk_rope(qkv, gamma, cos_rows, exact_proj=False)


def _cmp(name, got, ref, rel_tol=2e-3, assert_ok=True):
    diff = (got.float() - ref.float()).abs()
    rel = (diff.norm() / ref.float().norm().clamp_min(1e-6)).item()
    # ``cos`` is cosmetic past ~1e7 elements: float32 accumulation there reports
    # 1.001009 for an *exact* bf16 rounding of the reference (measured at
    # T=8192), so ``rel`` is the assertion and ``cos`` only catches gross
    # breakage.  A bf16 output cannot do better than rel ~1.7e-3.
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    ok = cos > 0.999 and rel < rel_tol
    verdict = ("ok" if ok else "FAIL") if assert_ok else f"rel<{rel_tol:g}: {ok}"
    print(f"  {name:<30}{verdict:<14}rel={rel:.2e} cos={cos:.6f} "
          f"maxabs={diff.max().item():.4f}")
    return ok


def _run(kernel, t_tok, block_m=BLOCK_M):
    dev = f"tpu:{_logical_device()}"
    torch.manual_seed(0)
    tp = -(-t_tok // block_m) * block_m
    # Every token gets its own slot, so large T needs a large pool; a repeat
    # would let one token's scatter clobber another's and fail the row check.
    # One slot past them is reserved as the pad slot.
    n_slots = max(SLOTS_TOTAL, t_tok + 1)
    s2 = n_slots * 2 * HKV
    pad_slot = n_slots - 1  # the pool's reserved row, excluded from checks

    x = torch.randn(t_tok, H).to(torch.bfloat16)
    w = (torch.randn(NTOT, H) / H**0.5).to(torch.bfloat16)
    gin = (torch.randn(1, H) * 0.1 + 1.0).to(torch.bfloat16)
    gamma = (torch.randn(2, D) * 0.5 + 1.0).float()
    positions = torch.randint(0, 4096, (t_tok,))
    cos_rows = _cos_sin_rows(positions)

    slots = torch.randperm(n_slots - 1)[:t_tok].int()
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
    # The driver's one-time fold of the input gamma into the projection weight.
    w_folded = (w.float() * gin.reshape(1, -1).float()).to(torch.bfloat16)

    # ``QOut`` is the PrimFunc's last param, so it is the only output and comes
    # back allocated; the caches are written in place through their pointers.
    q_dev = kernel(
        x_src.to(dev),
        w_folded.to(dev),
        cos_src.to(dev),
        gamma.to(dev),
        kv_row.to(dev),
        kc_dev,
        vc_dev,
    )
    q_out, kc_out, vc_out = q_dev.float().cpu(), kc_dev.float().cpu(), vc_dev.float().cpu()

    # Reassemble the two scattered halves into [T, HKV, D].
    def _halves(buf):
        return torch.stack(
            [torch.cat([buf[flats[:, 2 * h]], buf[flats[:, 2 * h + 1]]], dim=-1)
             for h in range(HKV)], dim=1,
        )

    k_got, v_got = _halves(kc_out), _halves(vc_out)

    ok = True
    q_ref, k_ref, v_ref = _ref_kernel_exact(x, w_folded, gamma, cos_rows)
    print("  -- vs the kernel's own algebra (pre-folded W, fp32 post-scale)")
    ok &= _cmp("q", q_out[:t_tok], q_ref)
    ok &= _cmp("k (scattered rows)", k_got, k_ref)
    ok &= _cmp("v (scattered rows)", v_got, v_ref)

    q_m, k_m, v_m = _ref_model(x, w, gin, gamma, cos_rows)
    print("  -- vs the torch model chain (not asserted; the rounding the")
    print("     fusion trades for one launch)")
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
    has_pad = t_tok % block_m != 0
    pad_ok = pad_written == has_pad
    print(f"  {'pad slot written (expect %s)' % has_pad:<30}"
          f"{'ok' if pad_ok else 'FAIL':<14}slot={pad_slot}")
    return ok and pad_ok


def _bench(kernel, t_tok, block_m):
    """Mean launch time of ``kernel`` over ``t_tok`` rows."""
    import time

    dev = f"tpu:{_logical_device()}"
    torch.manual_seed(0)
    tp = -(-t_tok // block_m) * block_m
    n_slots = max(SLOTS_TOTAL, t_tok + 1)
    s2 = n_slots * 2 * HKV
    pad_slot = n_slots - 1

    x = torch.randn(t_tok, H).to(torch.bfloat16)
    w = (torch.randn(NTOT, H) / H**0.5).to(torch.bfloat16)
    gin = (torch.randn(1, H) * 0.1 + 1.0).to(torch.bfloat16)
    gamma = (torch.randn(2, D) * 0.5 + 1.0).float()
    cos_rows = _cos_sin_rows(torch.randint(0, 4096, (t_tok,)))
    w_folded = (w.float() * gin.reshape(1, -1).float()).to(torch.bfloat16)

    # Slots may repeat here: this path measures time and checks ``q`` only, and
    # ``q`` does not depend on where the scatter lands.
    slots = torch.randint(0, n_slots - 1, (t_tok,)).int()
    flats = slots.long().unsqueeze(1) * (2 * HKV) + torch.arange(2 * HKV).long()
    kv_row = torch.empty(2 * HKV, tp, dtype=torch.uint32)
    kv_row[:, :t_tok] = flats.t()
    kv_row[:, t_tok:] = pad_slot * (2 * HKV) + torch.arange(2 * HKV).view(-1, 1)

    x_src = torch.zeros(tp, H).to(torch.bfloat16)
    x_src[:t_tok] = x
    cos_src = torch.zeros(tp, D)
    cos_src[:t_tok] = cos_rows
    kc = torch.zeros(s2, HALF).to(torch.bfloat16).to(dev)
    vc = torch.zeros(s2, HALF).to(torch.bfloat16).to(dev)
    args = (
        x_src.to(dev),
        w_folded.to(dev),
        cos_src.to(dev),
        gamma.to(dev),
        kv_row.to(dev),
        kc,
        vc,
    )

    q = kernel(*args)
    ref = _ref_kernel_exact(x, w_folded, gamma, cos_rows)[0]
    cos = torch.nn.functional.cosine_similarity(
        q.float().cpu()[:t_tok].flatten(), ref.float().flatten(), dim=0
    ).item()

    torch.tpu.synchronize()
    start = time.perf_counter()
    for _ in range(REPS):
        kernel(*args)
    torch.tpu.synchronize()
    return (time.perf_counter() - start) / REPS * 1e3, cos


def _sweep(block_ms, ts):
    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")
    for t_tok in ts:
        print(f"\n=== T={t_tok} ===")
        for bm in block_ms:
            label = f"fused_qkv_rope_ln {bm}"
            try:
                kernel = fused_qkv_rope_ln(block_M=bm, dtype="bfloat16", ln_eps=LN_EPS)
            except Exception as exc:  # local-memory 0x11, bad tile, ...
                print(f"  {label:<26}BUILD FAILED: {type(exc).__name__}: {exc}")
                continue
            try:
                t_ms, cos = _bench(kernel, t_tok, bm)
            except Exception as exc:
                print(f"  {label:<26}LAUNCH FAILED: {type(exc).__name__}: {exc}")
                continue
            print(
                f"  {label:<26}{t_ms:8.3f} ms   "
                f"cos={cos:.6f}{'' if cos > 0.999 else '   <-- WRONG'}"
            )


def main():
    if "--sweep" in sys.argv:
        arg = sys.argv[sys.argv.index("--sweep") + 1]
        spec = [int(v) for v in arg.split(",")]
        ts = [int(t) for t in sys.argv[sys.argv.index("--t") + 1].split(",")]
        print(f"sweep block_M={spec} at T={ts}")
        _sweep(spec, ts)
        return 0

    do_run = "--run" in sys.argv

    # Two launches: T on a tile boundary and T one tile short of one, which is
    # what exercises the reserved-slot padding path.
    kernels = {
        t: fused_qkv_rope_ln(block_M=BLOCK_M, dtype="bfloat16", ln_eps=LN_EPS)
        for t in (128, 100)
    }
    if not do_run:
        print("compiled (pass --run to execute)")
        return 0

    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")

    ok = True
    for t_tok, kernel in kernels.items():
        print(f"== T={t_tok}  (padded to {-(-t_tok // BLOCK_M) * BLOCK_M} rows)")
        ok &= _run(kernel, t_tok)
    print("PASS: fused ln+qkv/rope matches reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
