"""Standalone test + tuning harness for the fused ``gate_up -> silu_mul -> down`` kernel.

Compiles the **real tilelang DSL kernel** ``fused_mlp`` (see
:mod:`tilelang.tpu.kernels.fused_mlp`) through the generic TIR->PPL translator
and checks it against a plain torch reference of the whole MLP::

    X   [Tp, H]      bf16   normalized hidden (post_attention_layernorm applied)
    Wgu [2I, H]      bf16   gate_up_proj.weight, gate rows first
    Wd  [H, I]       bf16   down_proj.weight
    Y   [Tp, H]      bf16   output; rows past the token count are discarded

The reference computes the two projections and the silu in fp32 and rounds the
silu product to bf16 before the down projection -- the same rounding point the
kernel has (its one narrowing, ``T.copy(mid_f, mid)``).  A second, "model"
reference runs ``F.silu`` on the bf16 intermediate like ``SiluAndMul`` does; it is
printed, not asserted.

Usage (see the ``ppl-tpu-run-recipe`` memo for the env)::

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_mlp.py          # compile
    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_mlp.py --run    # correctness
    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_mlp.py --sweep  # tuning

``--sweep`` takes an optional spec: ``--sweep 128/128/256,256/128/256`` (or the
env ``SWEEP``), and ``--t 1024,8192`` (or ``TS``).  Each config is timed against
the same three torch ops on the same device and shapes, so the comparison is
live rather than remembered.
"""

import os
import sys
import time

import torch

from tilelang.tpu.kernels.fused_mlp import fused_mlp

H, I = 1024, 3072
BLOCK_SPEC = (128, 128, 256)  # block_M / block_I / block_K for the default build
REPS = 10
# block_M ladder first: it is the knob that sets the weight-stream traffic.  The
# full space is only walked when asked for.
SWEEP_DEFAULT = "64/128/256,128/128/256,256/128/256,512/128/256,1024/128/256"
SWEEP_TS_DEFAULT = "1024,8192"


def _logical_device() -> int:
    """The torch *logical* device id to run on (see ``test_flash_attention_paged``)."""
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


def _ref_fp32(x, wgu, wd):
    """What the kernel computes, in fp32, with the kernel's one bf16 rounding."""
    xf = x.float()
    gu = xf @ wgu.float().t()
    gate, up = gu[:, :I], gu[:, I:]
    mid = (gate / (1.0 + torch.exp(-gate))) * up
    return (mid.to(torch.bfloat16).float() @ wd.float().t()).to(torch.bfloat16)


def _ref_model(x, wgu, wd):
    """What ``gate_up_proj`` + ``SiluAndMul.forward_native`` + ``down_proj`` do."""
    gu = (x.float() @ wgu.float().t()).to(torch.bfloat16).float()
    gate, up = gu[:, :I], gu[:, I:]
    mid = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
    return (mid.float() @ wd.float().t()).to(torch.bfloat16)


def _cmp(name, got, ref, rel_tol=2e-3, assert_ok=True):
    diff = (got.float() - ref.float()).abs()
    rel = (diff.norm() / ref.float().norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    ok = cos > 0.999 and rel < rel_tol
    print(
        f"  {name:<30}{('ok' if ok else 'FAIL' if assert_ok else 'diff'):<14}"
        f"rel={rel:.2e} cos={cos:.6f} maxabs={diff.max().item():.4f}"
    )
    return ok or not assert_ok


def _inputs(t_tok, block_m, seed=0):
    """Padded device inputs; rows past ``t_tok`` are zero."""
    dev = f"tpu:{_logical_device()}"
    torch.manual_seed(seed)
    tp = -(-t_tok // block_m) * block_m
    x = torch.zeros(tp, H).to(torch.bfloat16)
    x[:t_tok] = torch.randn(t_tok, H).to(torch.bfloat16)
    wgu = (torch.randn(2 * I, H) / H**0.5).to(torch.bfloat16)
    wd = (torch.randn(H, I) / I**0.5).to(torch.bfloat16)
    return x[:t_tok], x.to(dev), wgu.to(dev), wd.to(dev), tp


def _run(kernel, t_tok, block_m):
    """Correctness at one token count (on and off a tile boundary)."""
    print(f"== T={t_tok}  (padded to {-(-t_tok // block_m) * block_m} rows)")
    x, x_dev, wgu_dev, wd_dev, tp = _inputs(t_tok, block_m)
    dev = f"tpu:{_logical_device()}"
    y = torch.empty(tp, H, dtype=torch.bfloat16, device=dev)
    kernel(x_dev, wgu_dev, wd_dev, y)
    y_out = y[:t_tok].float().cpu()
    ok = _cmp("fused_mlp", y_out, _ref_fp32(x, wgu_dev.cpu(), wd_dev.cpu()))
    # Not asserted: the bf16 chain rounds gate/up and the silu product to bf16,
    # the kernel keeps both in fp32 accumulators, so ~4e-3 (~bf16 eps) is the
    # *reference's* noise.  Printed to show which side is tighter.
    _cmp(
        "  vs bf16 torch chain",
        y_out,
        _ref_model(x, wgu_dev.cpu(), wd_dev.cpu()),
        assert_ok=False,
    )
    # Padded rows are read/written but must not affect real rows: re-running with
    # a different garbage tail has to produce the same first t_tok rows.
    y2 = torch.empty(tp, H, dtype=torch.bfloat16, device=dev)
    x_dev2 = x_dev.clone()
    if tp > t_tok:
        x_dev2[t_tok:] = torch.randn(tp - t_tok, H).to(torch.bfloat16).to(dev)
    kernel(x_dev2, wgu_dev, wd_dev, y2)
    same = torch.equal(y[:t_tok], y2[:t_tok])
    print(f"  {'pad rows isolated':<30}{('ok' if same else 'FAIL')}")
    return ok and same


def _time(fn):
    fn()
    torch.tpu.synchronize()
    start = time.perf_counter()
    for _ in range(REPS):
        fn()
    torch.tpu.synchronize()
    return (time.perf_counter() - start) / REPS * 1e3


def _torch_baseline(x_dev, wgu_dev, wd_dev, y=None):
    """The three launches the fusion replaces, timed together and apart."""
    def gate_up():
        return torch.nn.functional.linear(x_dev, wgu_dev)

    gu = gate_up()
    gate, up = gu[:, :I], gu[:, I:]

    def silu_mul():
        return torch.nn.functional.silu(gate) * up

    def down(mid):
        return torch.nn.functional.linear(mid, wd_dev)

    mid = silu_mul()
    t_gu = _time(gate_up)
    t_sm = _time(silu_mul)
    t_dn = _time(lambda: down(mid))

    def chain():
        g = gate_up()
        return down(torch.nn.functional.silu(g[:, :I]) * g[:, I:])

    t_all = _time(chain)
    print(
        f"  {'torch 3 ops':<22}{t_all:8.3f} ms   "
        f"(gate_up {t_gu:.3f} + silu_mul {t_sm:.3f} + down {t_dn:.3f}, "
        f"sum {t_gu + t_sm + t_dn:.3f})"
    )
    return t_all


def _sweep(spec, ts):
    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")
    for t_tok in ts:
        print(f"\n=== T={t_tok} ===")
        # Baseline first: same device state for every fused config below.
        _, x_dev, wgu_dev, wd_dev, _ = _inputs(t_tok, 64)
        base = _torch_baseline(x_dev, wgu_dev, wd_dev)
        for bm, bi, bk in spec:
            if I % bi or H % bk:
                print(f"  {bm}/{bi}/{bk:<12} skipped (geometry)")
                continue
            label = f"fused {bm}/{bi}/{bk}"
            try:
                kernel = fused_mlp(
                    block_M=bm, block_I=bi, block_K=bk, dtype="bfloat16"
                )
            except Exception as exc:  # local-memory 0x11, bad tile, ...
                print(f"  {label:<22}BUILD FAILED: {type(exc).__name__}: {exc}")
                continue
            x_pad, xd, wgu, wd, tp = _inputs(t_tok, bm)
            y = torch.empty(tp, H, dtype=torch.bfloat16, device=f"tpu:{_logical_device()}")
            try:
                t_fused = _time(lambda: kernel(xd, wgu, wd, y))
            except Exception as exc:
                print(f"  {label:<22}LAUNCH FAILED: {type(exc).__name__}: {exc}")
                continue
            got = y[:t_tok].float().cpu()
            ref = _ref_fp32(x_pad, wgu.cpu(), wd.cpu())
            cos = torch.nn.functional.cosine_similarity(
                got.flatten(), ref.float().flatten(), dim=0
            ).item()
            print(
                f"  {label:<22}{t_fused:8.3f} ms   {base / t_fused:5.2f}x vs torch   "
                f"cos={cos:.6f}{'' if cos > 0.999 else '   <-- WRONG'}"
            )


def main():
    do_run = "--run" in sys.argv
    if "--sweep" in sys.argv:
        arg = sys.argv[sys.argv.index("--sweep") + 1] if len(sys.argv) > sys.argv.index("--sweep") + 1 else ""
        spec_s = arg or os.environ.get("SWEEP", SWEEP_DEFAULT)
        ts_s = ""
        if "--t" in sys.argv:
            ts_s = sys.argv[sys.argv.index("--t") + 1]
        ts_s = ts_s or os.environ.get("TS", SWEEP_TS_DEFAULT)
        spec = [tuple(int(v) for v in c.split("/")) for c in spec_s.split(",")]
        ts = [int(t) for t in ts_s.split(",")]
        print(f"sweep {spec} at T={ts}")
        _sweep(spec, ts)
        return 0

    bm, bi, bk = BLOCK_SPEC
    kernel = fused_mlp(block_M=bm, block_I=bi, block_K=bk, dtype="bfloat16")
    if not do_run:
        print("compiled (pass --run to execute, --sweep to tune)")
        return 0

    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")

    ok = True
    for t_tok in (256, 200):
        ok &= _run(kernel, t_tok, bm)
    print("PASS: fused mlp matches reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
