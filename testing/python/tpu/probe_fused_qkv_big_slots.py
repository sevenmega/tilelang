"""Scale probe for the fused qkv/rope kernel's in-cache scatter.

``test_fused_qkv_rope.py`` runs against a 256-slot cache.  The real pool hands
the kernel ~10^6 slots, so the destination descriptor addresses several GB of
flat rows -- well past the 2^31-byte point where a 32-bit byte offset wraps.
That size is the one property of the production launch that the functional test
does not reproduce, and the kernel hung there, so this probe reruns the *same*
kernel at real cache sizes.

Two sizes are launched in one process, ascending, each printed before its
launch so a hang still leaves the smaller result on the record:

* ``--slots 200000``  -> ~0.4 GB of rows (below any 32-bit byte-offset limit)
* ``--slots 1400000`` -> ~2.9 GB of rows (above it; a full-size pool)

The checks are stronger than "the addressed rows are right", because a
misdirected write would still leave them right:

* the addressed rows match the torch reference, and
* ``cache.sum()`` equals the sum of exactly those rows -- so *nothing else* was
  written (the cache starts at zero, so the sum counts every stray write), and
* the reserved pad slot is all zeros.

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/probe_fused_qkv_big_slots.py
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from tilelang.tpu.kernels.fused_qkv_rope import fused_qkv_rope
from test_fused_qkv_rope import _cos_sin_rows, _ref

H, HQ, HKV, D = 1024, 16, 8, 128
HALF = D // 2
NTOT = (HQ + 2 * HKV) * D
EPS = 1e-6


def _rel(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-6)).item()


def _run(block_m: int, slots_total: int, t_tok: int) -> bool:
    dev = f"tpu:{torch.tpu.current_device()}"
    tp = -(-t_tok // block_m) * block_m
    s2 = slots_total * 2 * HKV
    pad_slot = slots_total - 1

    torch.manual_seed(0)
    x = torch.randn(t_tok, H).to(torch.bfloat16)
    w = (torch.randn(NTOT, H) / H**0.5).to(torch.bfloat16)
    gamma = torch.randn(2, D) * 0.5 + 1.0
    positions = torch.randint(0, 4096, (t_tok,))
    cos_rows = _cos_sin_rows(positions)

    # A sparse index table: token t -> slot t, so the addressed rows are easy to
    # name; padded columns all collapse onto the reserved tail slot, exactly as
    # the driver does.
    flats = (torch.arange(t_tok).view(-1, 1) * (2 * HKV)
             + torch.arange(2 * HKV)).long()                   # [T, 2*HKV], int64
    kv_row = torch.full((2 * HKV, tp), pad_slot * (2 * HKV), dtype=torch.int32)
    kv_row[:, :t_tok] = flats.t().to(torch.int32)

    x_src = torch.zeros(tp, H).to(torch.bfloat16)
    x_src[:t_tok] = x
    cos_src = torch.zeros(tp, D)
    cos_src[:t_tok] = cos_rows

    # Zero-filled: every stray write shows up in the sum below.
    kc = torch.zeros(s2, HALF, dtype=torch.bfloat16, device=dev)
    vc = torch.zeros_like(kc)

    print(f"  slots={slots_total} s2={s2} cache={kc.numel() * 2 / 2**30:.2f} GiB"
          f"  launching...", flush=True)
    t0 = time.perf_counter()
    q = kernel_call(
        x_src.to(dev), w.to(dev), cos_src.to(dev), gamma.to(dev),
        kv_row.to(dev), kc, vc, block_m,
    )
    q = q.float().cpu()
    print(f"  done in {time.perf_counter() - t0:.2f}s", flush=True)

    q_ref, k_ref, v_ref = _ref(x, w, gamma, cos_rows, True)

    ok = True
    ok &= _check("q", q[:t_tok], q_ref)
    for name, buf, ref in (("k", kc, k_ref), ("v", vc, v_ref)):
        got = torch.stack(
            [torch.cat([buf[flats[:, 2 * h]], buf[flats[:, 2 * h + 1]]], dim=-1)
             for h in range(HKV)],
            dim=1,
        ).float().cpu()
        ok &= _check(f"{name} (scattered rows)", got, ref)
        # Sum over the whole cache in fp32, on device: equals the sum of the
        # addressed rows iff nothing else was touched.
        total = buf.to(torch.float32).sum().item()
        want = ref.float().sum().item()
        good = abs(total - want) <= 1e-3 * max(abs(want), 1.0)
        print(f"  {name} cache sum {total:.4f} vs {want:.4f}"
              f"  {'ok' if good else 'FAIL (stray writes)'}", flush=True)
        ok &= good
    pad_rows = pad_slot * (2 * HKV) + torch.arange(2 * HKV)
    pad_zero = kc[pad_rows].sum().item() == 0
    print(f"  pad slot zeros {'ok' if pad_zero else 'FAIL'}", flush=True)
    return ok and pad_zero


def _check(name, got, ref):
    rel = _rel(got, ref)
    ok = rel < 2e-3
    print(f"  {name:<24}{'ok' if ok else 'FAIL':<6}rel={rel:.2e}", flush=True)
    if not ok:
        flat = got.float().flatten()
        print(f"    got[0,:4]={flat[:4].tolist()} ref[0,:4]="
              f"{ref.float().flatten()[:4].tolist()} nonzero={int((flat != 0).sum())}"
              f"/{flat.numel()}", flush=True)
    return ok


_cache: dict[int, object] = {}


def kernel_call(x, w, cos, gamma, kv_row, kc, vc, block_m):
    fn = _cache.get(block_m)
    if fn is None:
        fn = fused_qkv_rope(block_M=block_m, dtype="bfloat16")
        _cache[block_m] = fn
    return fn(x, w, cos, gamma, kv_row, kc, vc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, nargs="+", default=[200_000, 1_400_000])
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--block-m", type=int, default=16)
    args = ap.parse_args()

    import torch_tpu  # noqa: F401

    dev_id = int(__import__("os").environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])
    torch.tpu.set_device(dev_id)
    print(f"logical device {torch.tpu.current_device()}", flush=True)

    ok = True
    for slots in args.slots:
        print(f"== slots={slots}", flush=True)
        ok &= _run(args.block_m, slots, args.tokens)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
