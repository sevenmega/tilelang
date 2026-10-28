"""``o_proj`` + residual add as one tilelang DSL kernel for TPU.

The torch chain it replaces is one aten GEMM plus four eager elementwise
kernels::

    y = attn_output @ W_o^T                    # [T, H] -> DRAM
    h = (y.float() + residual.float()).to(bf16)  # _fused_ln_residual

At T=8192 measured on SG2260E, the GEMM is ~2.07 ms and the residual dance
~1.65 ms -- the four-op dance costs most of what the GEMM does, because it is
three full ``[T, H]`` passes plus two fp32 temporaries with no graph fusion in
between.  Here the residual seeds the fp32 accumulator, so ``h`` leaves in one
narrowing::

    h = A @ W^T + R

``h`` is the raw post-attention sum, which is exactly what the norm-fused MLP
(``fused_mlp_ln``) consumes *and* the value the layer carries forward as its
residual.  This kernel is therefore only meaningful when that MLP is in use;
``prepare_mlp`` would add the residual a second time.

Why the residual seeds the accumulator
--------------------------------------
The add belongs on the fp32 accumulator, before the one narrowing -- that is
both the cheapest placement and the most accurate.  It could also be written as
a ``T.Parallel`` pass after the k loop, but that needs a second ``[block_M, H]``
fp32 tile (64 KB/lane at ``block_M=1024``), while seeding reuses the tile the
GEMM is about to fill.  Seeding is safe where an elementwise op between a weight
load and its ``fmm2`` is not (see ``fused_qkv_rope_ln``): it happens *before*
the first weight load, and an ``fmm2`` accumulator is read-modify-write by
construction -- that is what lets a k sweep accumulate across iterations, and
the correctness test here shows the seed surviving into the result.

The one numerical difference from the torch chain: ``y`` is never rounded to
bf16, so the residual is added to the unrounded fp32 projection.  That is one
rounding fewer, and it is why greedy decoding can flip on a near tie.

Geometry
--------
``TP`` (token rows) is padded to a multiple of ``block_M`` by the driver, as in
``fused_mlp``; padded rows read garbage from ``A`` and ``R`` and are dropped by
the driver's slice -- each row's output depends only on that row.

``A`` is the attention output ``[T, Hq*D]``, ``W`` is ``o_proj.weight`` as
stored (``[H, Hq*D]``, consumed through ``transpose_B=True`` -- no prepack), and
``R`` is the residual.  ``W`` is streamed in ``block_K``-wide slabs of its
*contraction* axis, once per row tile; as in ``fused_mlp`` the four cores walk
the same weight sequence in lockstep and share one stream.

Integer-width hazards this file obeys (both documented in ``fused_mlp_ln``):
the emitter resolves a ``BufferLoad`` of a local tile to the *whole tile name*
and ignores its indices, so no tile here is ever sliced -- there is exactly one
output tile whose width is ``H``, not one per column chunk.  And PPL has no
implicit bf16->fp32 widening; the residual cast is spelled ``T.Cast``.
"""

import tilelang
import tilelang.language as T

from tilelang.tpu.op.gemm.tiled import gemm_tiled

# Dynamic extents.  ``TP`` is the token count padded to ``block_M``; everything
# else -- the model geometry and the tile sizes -- is baked, so one build serves
# one model at one tile size and every batch shape.
TP = T.dynamic("TP", "int32")

# SG2260E has four cores; the last grid axis is the core index. Baked.
_NUM_CORES = 4


@tilelang.jit(target="tpu")
def oproj_res(
    hidden: int = 1024,
    input_size: int = 2048,
    block_M: int = 128,
    block_K: int = 256,
    dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
):
    """Attention output projection with the residual folded into the accumulator.

    ``block_M`` is the row tile (the cost knob: a row tile streams the whole
    ``H x input_size`` weight) and ``block_K`` the reduction tile over
    ``input_size``.  The output width is ``hidden``, whole: see the module
    docstring on why it is not chunked.
    """
    H = int(hidden)
    K = int(input_size)
    assert K % block_K == 0, "input_size must be a multiple of block_K"
    k_steps = K // block_K

    @T.prim_func
    def main(
        A: T.Tensor((TP, K), dtype),
        W: T.Tensor((H, K), dtype),
        R: T.Tensor((TP, H), dtype),
        Y: T.Tensor((TP, H), dtype),
    ):
        Tp, _ = A.shape
        Mt = T.ceildiv(Tp, block_M)
        tiles_per_core = T.ceildiv(Mt, _NUM_CORES)

        with T.Kernel(Mt, 1, _NUM_CORES) as (bx, by, bc):
            a_t = T.alloc_local((block_M, block_K), dtype)
            w_t = T.alloc_local((H, block_K), dtype)
            # Landing tile for the residual: it is bf16 in memory and the
            # accumulator is fp32, and PPL has no implicit widening, so the copy
            # needs a bf16 destination before the explicit ``T.Cast``.
            r_t = T.alloc_local((block_M, H), dtype)
            # The whole output row tile, live across the k sweep only.
            acc = T.alloc_local((block_M, H), accum_dtype)

            g = bc * tiles_per_core + bx
            with T.If(g < Mt), T.Then():
                row0 = g * block_M

                # The residual seeds the accumulator: h = A @ W^T + R with the
                # add inside the fp32 accumulation, before any weight is loaded.
                T.copy(R[row0, 0], r_t)
                for i, j in T.Parallel(block_M, H):
                    acc[i, j] = T.Cast("float32", r_t[i, j])

                for kb in T.serial(k_steps):
                    k0 = kb * block_K
                    T.copy(A[row0, k0], a_t)
                    T.copy(W[0, k0], w_t)
                    gemm_tiled(a_t, w_t, acc, transpose_B=True)

                # The one narrowing.
                T.copy(acc, Y[row0, 0])

    return main
