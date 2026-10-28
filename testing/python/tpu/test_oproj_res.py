"""Standalone test + tuning harness for the fused ``o_proj`` + residual kernel.

Compiles the **real tilelang DSL kernel** ``oproj_res`` (see
:mod:`tilelang.tpu.kernels.oproj_res`) and checks it against a plain torch
reference::

    A  [Tp, K]   bf16   attention output, K = Hq*D (2048 for Qwen3-0.6B)
    W  [H, K]    bf16   o_proj.weight, consumed as stored
    R  [Tp, H]   bf16   residual
    Y  [Tp, H]   bf16   h = A @ W^T + R; rows past the token count are discarded

The reference adds the residual to the **unrounded** fp32 projection, which is
what "the residual seeds the accumulator" means.  A second, "model" reference
rounds the projection to bf16 first (today's ``mm`` -> ``_fused_ln_residual``
chain) and is printed, not asserted -- that rounding is the one the kernel
deletes.

The first job of this file is **pricing**, not correctness: the boundary it
replaces is an aten GEMM plus a four-op residual dance, and the dance appears in
no profile row.  ``--sweep`` prints both sides live so the comparison is
measured on the same device, at the same shapes, in one run.

Usage (see the ``ppl-tpu-run-recipe`` memo for the env)::

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_oproj_res.py          # compile
    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_oproj_res.py --run    # correctness
    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_oproj_res.py --sweep  # tuning

``--sweep`` takes an optional ``block_M/block_K`` spec (or the env ``SWEEP``) and
``--t 1024,8192`` (or ``TS``).  One spec per process -- see ``SWEEP_DEFAULT``.
"""

import os
import sys
import time

import torch

from tilelang.tpu.kernels.oproj_res import oproj_res

H, K = 1024, 2048
BLOCK_SPEC = (1024, 1024)  # block_M / block_K for the default build
REPS = 10
# block_M sets the weight-stream traffic, so it leads; block_K is the reduction
# tile over `input_size`.  One config per process: a *second* config in the same
# process never returns from ``py_init_device_shared`` on this runtime -- the
# chip sits at 0% util while the run blocks, and killing the stuck task can
# leave the driver needing a reload.  Sweeping more sizes means more processes,
# one spec each (``--allow-multi`` overrides, for someone studying the hazard).
SWEEP_DEFAULT = "1024/1024"
SWEEP_TS_DEFAULT = "1024,8192"


def _logical_device() -> int:
    """The torch *logical* device id to run on (see ``test_flash_attention_paged``)."""
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


def _dev() -> str:
    return f"tpu:{_logical_device()}"


def _ref_fp32(a, w, r):
    """What the kernel computes: the add on the unrounded fp32 projection."""
    return ((a.float() @ w.float().t()) + r.float()).to(torch.bfloat16)


def _ref_model(a, w, r):
    """What today's chain does: round the projection to bf16, then add."""
    y = (a.float() @ w.float().t()).to(torch.bfloat16)
    return (y.float() + r.float()).to(torch.bfloat16)


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
    """Padded device inputs; rows past ``t_tok`` are zero.

    Scale is chosen so a bf16 output lands around unit magnitude, as it does in
    the model (``o_proj`` sees an attention output already in hidden-state
    scale).
    """
    torch.manual_seed(seed)
    tp = -(-t_tok // block_m) * block_m
    a = torch.zeros(tp, K).to(torch.bfloat16)
    r = torch.zeros(tp, H).to(torch.bfloat16)
    a[:t_tok] = (torch.randn(t_tok, K) / K**0.5).to(torch.bfloat16)
    r[:t_tok] = torch.randn(t_tok, H).to(torch.bfloat16)
    w = (torch.randn(H, K) / K**0.5).to(torch.bfloat16)
    return (a[:t_tok], r[:t_tok], a.to(_dev()), r.to(_dev()), w.to(_dev()), tp)


def _run(kernel, t_tok, block_m):
    """Correctness at one token count (on and off a tile boundary)."""
    print(f"== T={t_tok}  (padded to {-(-t_tok // block_m) * block_m} rows)")
    a, r, a_dev, r_dev, w_dev, tp = _inputs(t_tok, block_m)
    w_cpu = w_dev.cpu()
    ref = _ref_fp32(a, w_cpu, r)
    ref_model = _ref_model(a, w_cpu, r)
    y = torch.empty(tp, H, dtype=torch.bfloat16, device=_dev())
    kernel(a_dev, w_dev, r_dev, y)
    y_out = y[:t_tok].float().cpu()
    ok = _cmp("oproj_res", y_out, ref)
    # Not asserted: the bf16 rounding of the projection is a real difference in
    # today's chain, not noise in the reference.
    _cmp("  vs bf16-rounded chain", y_out, ref_model, assert_ok=False)

    # The residual has to arrive per row: a broadcast-bias bug (what
    # ``torch.addmm`` does on this stack) leaves every row's offset equal to row
    # 0's.  Compare the kernel's residual contribution against the reference one
    # row by row.
    off = y_out - (a.float() @ w_cpu.float().t()).to(torch.bfloat16)
    spread = (off - off[0:1]).abs().max(dim=1).values
    n_rows = int((spread > 1e-2).sum())
    per_row = n_rows > t_tok // 2
    print(f"  {'residual is per-row':<30}{('ok' if per_row else 'FAIL')}"
          f"            ({n_rows}/{t_tok} rows differ from row 0)")

    # Padded rows are read/written but must not affect real rows.
    y2 = torch.empty(tp, H, dtype=torch.bfloat16, device=_dev())
    a_dev2, r_dev2 = a_dev.clone(), r_dev.clone()
    if tp > t_tok:
        a_dev2[t_tok:] = (torch.randn(tp - t_tok, K) / K**0.5).to(torch.bfloat16).to(_dev())
        r_dev2[t_tok:] = torch.randn(tp - t_tok, H).to(torch.bfloat16).to(_dev())
    kernel(a_dev2, w_dev, r_dev2, y2)
    same = torch.equal(y[:t_tok], y2[:t_tok])
    print(f"  {'pad rows isolated':<30}{('ok' if same else 'FAIL')}")
    return ok and per_row and same


def _time(fn, reps=REPS):
    """Median serialized wall time in ms (device drained around every round)."""
    for _ in range(2):
        fn()
    torch.tpu.synchronize()
    ts = []
    for _ in range(reps):
        torch.tpu.synchronize()
        tic = time.perf_counter()
        fn()
        torch.tpu.synchronize()
        ts.append(time.perf_counter() - tic)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def _torch_baseline(a_dev, r_dev, w_dev):
    """The boundary the kernel replaces: the aten GEMM plus the residual dance."""
    def mm():
        return torch.mm(a_dev, w_dev.t())

    y = mm()

    def dance():
        return (y.float() + r_dev.float()).to(torch.bfloat16)

    def chain():
        return (torch.mm(a_dev, w_dev.t()).float() + r_dev.float()).to(torch.bfloat16)

    t_mm = _time(mm)
    t_dance = _time(dance)
    t_chain = _time(chain)
    print(
        f"  {'torch mm + dance':<22}{t_chain:8.3f} ms   "
        f"(mm {t_mm:.3f} + dance {t_dance:.3f}, sum {t_mm + t_dance:.3f})"
    )
    return t_chain


def _sweep(spec, ts):
    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")
    for t_tok in ts:
        print(f"\n=== T={t_tok} ===")
        # Baseline first: same device state for every fused config below.
        _, _, a_dev, r_dev, w_dev, _ = _inputs(t_tok, 64)
        t_torch = _torch_baseline(a_dev, r_dev, w_dev)
        a_cpu, r_cpu = a_dev[:t_tok].float().cpu(), r_dev[:t_tok].float().cpu()
        w_cpu = w_dev.cpu()
        ref = ((a_cpu @ w_cpu.float().t()) + r_cpu).to(torch.bfloat16)
        for bm, bk in spec:
            if K % bk:
                print(f"  {bm}/{bk:<16} skipped (K % block_K)")
                continue
            label = f"oproj_res {bm}/{bk}"
            _, _, a_dev, r_dev, w_dev, tp = _inputs(t_tok, bm)
            y = torch.empty(tp, H, dtype=torch.bfloat16, device=_dev())
            try:
                kernel = oproj_res(block_M=bm, block_K=bk, dtype="bfloat16")
            except Exception as exc:  # local-memory 0x11, bad tile, ...
                print(f"  {label:<22}BUILD FAILED: {type(exc).__name__}: {exc}")
                continue
            try:
                t_fused = _time(lambda: kernel(a_dev, w_dev, r_dev, y))
            except Exception as exc:
                print(f"  {label:<22}LAUNCH FAILED: {type(exc).__name__}: {exc}")
                continue
            got = y[:t_tok].float().cpu()
            cos = torch.nn.functional.cosine_similarity(
                got.flatten(), ref.float().flatten(), dim=0
            ).item()
            print(
                f"  {label:<22}{t_fused:8.3f} ms   "
                f"{t_torch / t_fused:5.2f}x vs torch   "
                f"cos={cos:.6f}{'' if cos > 0.999 else '   <-- WRONG'}"
            )


def main():
    do_run = "--run" in sys.argv
    if "--sweep" in sys.argv:
        i = sys.argv.index("--sweep")
        arg = sys.argv[i + 1] if len(sys.argv) > i + 1 else ""
        spec_s = arg or os.environ.get("SWEEP", SWEEP_DEFAULT)
        ts_s = ""
        if "--t" in sys.argv:
            ts_s = sys.argv[sys.argv.index("--t") + 1]
        ts_s = ts_s or os.environ.get("TS", SWEEP_TS_DEFAULT)
        spec = [tuple(int(v) for v in c.split("/")) for c in spec_s.split(",")]
        ts = [int(t) for t in ts_s.split(",")]
        if len(spec) > 1 and "--allow-multi" not in sys.argv:
            raise SystemExit(
                f"{len(spec)} configs in one process: only the first initializes "
                "on this runtime (see SWEEP_DEFAULT). Run one spec per process, "
                "or pass --allow-multi if that is what you are testing."
            )
        print(f"sweep {spec} at T={ts}")
        _sweep(spec, ts)
        return 0

    bm, bk = BLOCK_SPEC
    kernel = oproj_res(block_M=bm, block_K=bk, dtype="bfloat16")
    if not do_run:
        print("compiled (pass --run to execute, --sweep to tune)")
        return 0

    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")

    ok = True
    for t_tok in (256, 200):
        ok &= _run(kernel, t_tok, bm)
    print("PASS: fused o_proj + residual matches reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
