"""Standalone test + tuning harness for the fused ``gate_up -> silu_mul -> down`` kernel.

Compiles the **real tilelang DSL kernels** ``fused_mlp`` and ``fused_mlp_ln``
(see :mod:`tilelang.tpu.kernels.fused_mlp` / ``._ln``) through the generic
TIR->PPL translator and checks them against a plain torch reference::

    X   [Tp, H]      bf16   hidden; normalized for ``fused_mlp``, the raw
                            ``hidden + residual`` for ``fused_mlp_ln``
    Wgu [2I, H]      bf16   gate_up_proj.weight, gate rows first; for the ``_ln``
                            kernel it is pre-folded with the norm's gamma
    Wd  [H, I]       bf16   down_proj.weight
    Y   [Tp, H]      bf16   output; rows past the token count are discarded

The reference computes the two projections and the silu in fp32 and rounds the
silu product to bf16 before the down projection -- the same rounding point the
kernel has (its one narrowing, ``T.copy(mid_f, mid)``).  The ``_ln`` reference
additionally folds gamma into ``Wgu`` (rounded to bf16 first, as the driver does)
and applies ``rstd`` to the fp32 accumulators, i.e. the kernel's own algebra.  A
second, "model" reference runs ``F.silu`` on the bf16 intermediate like
``SiluAndMul`` does (and, for ``_ln``, normalizes the bf16 activation like
``RMSNorm.forward_native``); it is printed, not asserted.

Usage (see the ``ppl-tpu-run-recipe`` memo for the env)::

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_mlp.py          # compile
    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_mlp.py --run    # correctness
    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_fused_mlp.py --sweep  # tuning
    ... add ``--ln`` to any of the above to test the norm-folded kernel

``--sweep`` takes an optional spec: ``--sweep 128/128/256,256/128/256`` (or the
env ``SWEEP``), and ``--t 1024,8192`` (or ``TS``).  Each config is timed against
the same three torch ops on the same device and shapes, so the comparison is
live rather than remembered.  With ``--ln`` each config builds *both* kernels, so
the print reads as "what the fold costs against its sibling".
"""

import os
import sys
import time

import torch

from tilelang.tpu.kernels.fused_mlp import fused_mlp
from tilelang.tpu.kernels.fused_mlp_ln import fused_mlp_ln

H, I = 1024, 3072
BLOCK_SPEC = (1024, 256, 1024)  # block_M / block_I / block_K for the default build
LN_EPS = 1e-6
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


def _gamma(seed=0):
    """A driver-shaped norm gamma: signed, near one."""
    torch.manual_seed(seed + 1)
    return (torch.randn(1, H) * 0.1 + 1.0).to(torch.bfloat16)


def _fold(wgu, gin):
    """``gamma * Wgu`` rounded back to bf16 -- the driver's fold, exactly."""
    return (wgu.float() * gin.reshape(1, -1).float()).to(wgu.dtype)


def _rms(x, gin, eps):
    """``RMSNorm.forward_native`` on a bf16 activation, one rounding."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * rstd * gin.reshape(1, -1).float()).to(x.dtype)


def _ref_ln_fp32(x, wgu, wd, gin, eps):
    """What ``fused_mlp_ln`` computes: pre-folded W, rstd on the accumulators."""
    xf = x.float()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    gu = xf @ _fold(wgu, gin).float().t()
    gate, up = gu[:, :I] * rstd, gu[:, I:] * rstd
    mid = (gate / (1.0 + torch.exp(-gate))) * up
    return (mid.to(torch.bfloat16).float() @ wd.float().t()).to(torch.bfloat16)


def _ref_ln_model(x, wgu, wd, gin, eps):
    """What the torch chain would do: normalize the bf16 activation, then project."""
    xn = _rms(x, gin, eps)
    gu = (xn.float() @ wgu.float().t()).to(torch.bfloat16).float()
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


def _run(kernel, t_tok, block_m, ln=False):
    """Correctness at one token count (on and off a tile boundary)."""
    print(f"== T={t_tok}  (padded to {-(-t_tok // block_m) * block_m} rows)"
          f"{'  +ln' if ln else ''}")
    x, x_dev, wgu_dev, wd_dev, tp = _inputs(t_tok, block_m)
    dev = f"tpu:{_logical_device()}"
    wgu_cpu, wd_cpu = wgu_dev.cpu(), wd_dev.cpu()
    if ln:
        gin = _gamma()
        w_dev = _fold(wgu_cpu, gin).to(dev)
        ref = _ref_ln_fp32(x, wgu_cpu, wd_cpu, gin, LN_EPS)
        ref_model = _ref_ln_model(x, wgu_cpu, wd_cpu, gin, LN_EPS)
    else:
        w_dev = wgu_dev
        ref = _ref_fp32(x, wgu_cpu, wd_cpu)
        ref_model = _ref_model(x, wgu_cpu, wd_cpu)
    y = torch.empty(tp, H, dtype=torch.bfloat16, device=dev)
    kernel(x_dev, w_dev, wd_dev, y)
    y_out = y[:t_tok].float().cpu()
    ok = _cmp("fused_mlp_ln" if ln else "fused_mlp", y_out, ref)
    # Not asserted: the bf16 chain rounds gate/up and the silu product to bf16,
    # the kernel keeps both in fp32 accumulators, so ~4e-3 (~bf16 eps) is the
    # *reference's* noise.  Printed to show which side is tighter.
    _cmp("  vs bf16 torch chain", y_out, ref_model, assert_ok=False)
    # Padded rows are read/written but must not affect real rows: re-running with
    # a different garbage tail has to produce the same first t_tok rows.  With
    # the norm folded in this also guards the sum-of-squares pass, the only
    # cross-row shape in the kernel.
    y2 = torch.empty(tp, H, dtype=torch.bfloat16, device=dev)
    x_dev2 = x_dev.clone()
    if tp > t_tok:
        x_dev2[t_tok:] = torch.randn(tp - t_tok, H).to(torch.bfloat16).to(dev)
    kernel(x_dev2, w_dev, wd_dev, y2)
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


def _sweep(spec, ts, ln=False):
    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")
    gin = _gamma()
    for t_tok in ts:
        print(f"\n=== T={t_tok} ===")
        # Baseline first: same device state for every fused config below.
        _, x_dev, wgu_dev, wd_dev, _ = _inputs(t_tok, 64)
        base = _torch_baseline(x_dev, wgu_dev, wd_dev)
        if ln:
            g_dev = gin.to(x_dev.device)
            t_norm = _time(lambda: _rms(x_dev, g_dev, LN_EPS))
            print(f"  {'torch rmsnorm row':<22}{t_norm:8.3f} ms   (what the fold deletes)")
        # The ``_ln`` variant deletes the standalone norm row too, so it is
        # measured against the four-op chain it replaces, not the three the
        # plain variant does.
        torch_ln, torch_mlp = base + (t_norm if ln else 0.0), base
        for bm, bi, bk in spec:
            if I % bi or H % bk or H % bi:
                print(f"  {bm}/{bi}/{bk:<12} skipped (geometry)")
                continue
            x_pad, xd, wgu, wd, tp = _inputs(t_tok, bm)
            wgu_cpu, wd_cpu = wgu.cpu(), wd.cpu()
            y = torch.empty(tp, H, dtype=torch.bfloat16, device=f"tpu:{_logical_device()}")
            for variant in ([False, True] if ln else [False]):
                label = f"{'fused_ln' if variant else 'fused'} {bm}/{bi}/{bk}"
                try:
                    if variant:
                        kernel = fused_mlp_ln(
                            block_M=bm, block_I=bi, block_K=bk,
                            ln_eps=LN_EPS, dtype="bfloat16",
                        )
                        w = _fold(wgu_cpu, gin).to(xd.device)
                        ref = _ref_ln_fp32(x_pad, wgu_cpu, wd_cpu, gin, LN_EPS)
                    else:
                        kernel = fused_mlp(
                            block_M=bm, block_I=bi, block_K=bk, dtype="bfloat16"
                        )
                        w = wgu
                        ref = _ref_fp32(x_pad, wgu_cpu, wd_cpu)
                except Exception as exc:  # local-memory 0x11, bad tile, ...
                    print(f"  {label:<22}BUILD FAILED: {type(exc).__name__}: {exc}")
                    continue
                try:
                    t_fused = _time(lambda: kernel(xd, w, wd, y))
                except Exception as exc:
                    print(f"  {label:<22}LAUNCH FAILED: {type(exc).__name__}: {exc}")
                    continue
                got = y[:t_tok].float().cpu()
                cos = torch.nn.functional.cosine_similarity(
                    got.flatten(), ref.float().flatten(), dim=0
                ).item()
                print(
                    f"  {label:<22}{t_fused:8.3f} ms   "
                    f"{(torch_ln if variant else torch_mlp) / t_fused:5.2f}x vs torch   "
                    f"cos={cos:.6f}{'' if cos > 0.999 else '   <-- WRONG'}"
                )


def main():
    do_run = "--run" in sys.argv
    ln = "--ln" in sys.argv
    if "--sweep" in sys.argv:
        arg = sys.argv[sys.argv.index("--sweep") + 1] if len(sys.argv) > sys.argv.index("--sweep") + 1 else ""
        spec_s = arg or os.environ.get("SWEEP", SWEEP_DEFAULT)
        ts_s = ""
        if "--t" in sys.argv:
            ts_s = sys.argv[sys.argv.index("--t") + 1]
        ts_s = ts_s or os.environ.get("TS", SWEEP_TS_DEFAULT)
        spec = [tuple(int(v) for v in c.split("/")) for c in spec_s.split(",")]
        ts = [int(t) for t in ts_s.split(",")]
        print(f"sweep {spec} at T={ts}{'  (+ln)' if ln else ''}")
        _sweep(spec, ts, ln=ln)
        return 0

    bm, bi, bk = BLOCK_SPEC
    if ln:
        kernel = fused_mlp_ln(
            block_M=bm, block_I=bi, block_K=bk, ln_eps=LN_EPS, dtype="bfloat16"
        )
    else:
        kernel = fused_mlp(block_M=bm, block_I=bi, block_K=bk, dtype="bfloat16")
    if not do_run:
        print("compiled (pass --run to execute, --sweep to tune)")
        return 0

    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")

    ok = True
    for t_tok in (256, 200):
        ok &= _run(kernel, t_tok, bm, ln=ln)
    print("PASS: fused mlp matches reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
