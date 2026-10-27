"""Probe: the two instruction classes the fused input-norm variance pass needs.

The variance of a row (``rstd = 1/sqrt(mean(x^2)+eps)``) has to be accumulated in
fp32, but a tile loaded from a bf16 tensor is bf16 and ``tiu::fadd``/``tiu::fmul``
demand a single DataType across all three operands -- there is no implicit
widening.  Two ways out, probed here separately so a failure names the culprit:

* **A -- widen.**  ``Y = float32(X)``, elementwise.  The emitter turns
  ``T.Cast("float32", x[i, j])`` into ``tiu::cast(dst_fp32, src_bf16)``, the same
  instruction as the fp32->bf16 narrowing ``T.copy`` already uses.  (The
  ``tiu::fmul_cast(dst, src, 1.0f)`` spelling declared in ``ppl_tiu_func.h`` is
  rejected by ppl-compile as "unsupported arith type", so this is the only
  widening spelling available.)  If the widening ``cast`` is miscompiled rather
  than merely undocumented, this probe is where it shows: the reference is exact.
* **B -- widen, square, reduce.**  The whole variance pass of the fused kernel
  in isolation: widen each chunk to fp32, square, ``quick_pooling`` the row sum,
  and carry the chunk sums in a 2-D fp32 tile by a row broadcast.  If A is exact
  but the fused kernel still mis-normalizes, the fault is in how this pass
  shares local memory with its neighbours, not in the instructions.

Both are compared against exact torch arithmetic; B's tolerance is the bf16
square's, not the probe's.

Run on real hardware under the PPL env (see the ``ppl-tpu-run-recipe`` memo):

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/probe_cast_widen.py --run
"""

import os
import sys

import tilelang
import tilelang.language as T


def _logical_device() -> int:
    """The torch *logical* device id to run on (see ``test_flash_attention_paged``)."""
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


@tilelang.jit(target="tpu")
def widen_probe(rows: int = 64, width: int = 1024, block_K: int = 256):
    """``Y = float32(X)``: bf16 -> fp32 ``tiu::cast``, exact."""

    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.bfloat16),
        Y: T.Tensor((rows, width), T.float32),
    ):
        with T.Kernel(1, 1, 1) as (bx, by, bc):
            x = T.alloc_local((rows, block_K), T.bfloat16)
            y = T.alloc_local((rows, block_K), T.float32)
            for kk in T.serial(width // block_K):
                T.copy(X[0, kk * block_K], x)
                for i, j in T.Parallel(rows, block_K):
                    y[i, j] = T.Cast("float32", x[i, j])
                T.copy(y, Y[0, kk * block_K])

    return main


@tilelang.jit(target="tpu")
def sqsum_probe(rows: int = 64, width: int = 1024, block_K: int = 64):
    """``S[r] = sum_w X[r,w]^2``: square in bf16, pool into fp32, fadd the chunks."""

    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.bfloat16),
        S: T.Tensor((rows, block_K), T.float32),
    ):
        with T.Kernel(1, 1, 1) as (bx, by, bc):
            x = T.alloc_local((rows, block_K), T.bfloat16)
            y = T.alloc_local((rows, block_K), T.float32)
            part = T.alloc_local((rows,), T.float32)
            acc = T.alloc_local((rows, block_K), T.float32)
            T.fill(acc, 0)
            for kk in T.serial(width // block_K):
                T.copy(X[0, kk * block_K], x)
                # Widen first: quick_pooling insists on one DataType for source
                # and destination ("deduced conflicting types 'float' vs
                # 'bf16'"), so a bf16 square cannot be reduced into fp32.
                for i, j in T.Parallel(rows, block_K):
                    y[i, j] = T.Cast("float32", x[i, j])
                for i, j in T.Parallel(rows, block_K):
                    y[i, j] = y[i, j] * y[i, j]
                T.reduce_sum(y, part, dim=1)
                # Row broadcast (``part[i]`` has no j), the verified direction.
                for i, j in T.Parallel(rows, block_K):
                    acc[i, j] = acc[i, j] + part[i]
            T.copy(acc, S[0, 0])

    return main


@tilelang.jit(target="tpu")
def sqsum_acc_probe(rows: int = 64, width: int = 1024, block_K: int = 64):
    """The fused kernel's pass, byte for byte: accumulate tiles, reduce once.

    ``sqsum_probe`` reduces every chunk and carries the scalar sums; this carries
    the squares in a 2-D tile exactly as ``fused_qkv_rope_ln`` does, so a failure
    here names the accumulate-and-reduce-once structure.
    """

    @T.prim_func
    def main(
        X: T.Tensor((rows, width), T.bfloat16),
        S: T.Tensor((rows, block_K), T.float32),
    ):
        with T.Kernel(1, 1, 1) as (bx, by, bc):
            xs = T.alloc_local((rows, block_K), T.bfloat16)
            n_hi = T.alloc_local((rows, block_K), T.float32)
            n_lo = T.alloc_local((rows, block_K), T.float32)
            out = T.alloc_local((rows, block_K), T.float32)
            part = T.alloc_local((rows,), T.float32)
            T.fill(n_lo, 0)
            for kk in T.serial(width // block_K):
                T.copy(X[0, kk * block_K], xs)
                for i, j in T.Parallel(rows, block_K):
                    n_hi[i, j] = T.Cast("float32", xs[i, j])
                for i, j in T.Parallel(rows, block_K):
                    n_hi[i, j] = n_hi[i, j] * n_hi[i, j]
                for i, j in T.Parallel(rows, block_K):
                    n_lo[i, j] = n_lo[i, j] + n_hi[i, j]
            T.reduce_sum(n_lo, part, dim=1)
            for i, j in T.Parallel(rows, block_K):
                out[i, j] = part[i]
            T.copy(out, S[0, 0])

    return main


def _cmp(name: str, got, ref, rel_tol: float) -> bool:
    import torch

    diff = (got.float() - ref.float()).abs()
    rel = (diff.norm() / ref.float().norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.float().flatten(), ref.float().flatten(), dim=0
    ).item()
    ok = cos > 0.999 and rel < rel_tol
    print(f"  {name:<24}{'ok' if ok else 'FAIL':<8}rel={rel:.2e} cos={cos:.6f} "
          f"maxabs={diff.max().item():.4g}")
    return ok


def main(do_run: bool) -> int:
    rows, width = 64, 1024
    widen = widen_probe(rows=rows, width=width)
    sqsum = sqsum_probe(rows=rows, width=width)
    sqsum_acc = sqsum_acc_probe(rows=rows, width=width)
    if not do_run:
        print("compiled (pass --run to execute)")
        return 0

    import torch
    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")
    dev = f"tpu:{_logical_device()}"

    # Values spanning bf16's dynamic range, including exact and subnormal-ish
    # magnitudes: a widening cast that reinterprets bits instead of converting
    # shows up as wrong exponents, not as noise.
    torch.manual_seed(0)
    x = (torch.randn(rows, width) * torch.logspace(-3, 3, width)).to(torch.bfloat16)

    y_dev = widen(x.to(dev))  # last param -> allocated by the wrapper
    ok = _cmp("A widen", y_dev.cpu(), x.float(), rel_tol=0.0)

    s_dev = sqsum(x.to(dev))
    # Every column of a row holds that row's sum; column 0 is the answer.
    ok &= _cmp("B widen+square+reduce", s_dev.cpu()[:, 0], x.float().pow(2).sum(-1),
               rel_tol=2e-3)

    s2_dev = sqsum_acc(x.to(dev))
    ok &= _cmp("C accumulate-then-reduce", s2_dev.cpu()[:, 0],
               x.float().pow(2).sum(-1), rel_tol=2e-3)

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main("--run" in sys.argv))
