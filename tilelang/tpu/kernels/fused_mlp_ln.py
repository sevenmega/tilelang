"""``post_attention_layernorm -> gate_up_proj -> silu_mul -> down_proj``, TPU.

The fused MLP of ``fused_mlp`` with the layer's post-attention RMSNorm folded in,
the MLP-side twin of ``fused_qkv_rope_ln``.  The torch chain it replaces::

    x = RMSNorm(hidden + residual)        # [T, H] -> DRAM
    gate_up, _ = gate_up_proj(x)          # [T, 2I] -> DRAM
    mid        = silu(gate_up[..., :I]) * gate_up[..., I:]
    y, _       = down_proj(mid)           # [T, H]

At T=8192 the norm is a full bf16 read+write of ``[T, H]`` (~34 MB) and it is
memory bound, so almost the whole row is that round trip.  Here the kernel takes
the *un-normalized* sum, normalizes on chip, and the intermediate never leaves.

The norm folds in exactly
-------------------------
``x_norm = (x * rstd) * gamma`` and ``gate = x_norm . Wgu^T``, so::

    gate = rstd * (x . (gamma * Wgu)^T)

``gamma`` indexes the **contraction** axis of ``Wgu`` ``[2I, H]``, so it is a
per-column scale of the weight and folds into it on the driver -- one
``[2I, H]`` elementwise pass per layer, cached per weight version, exactly as
``TpuFusedQkvLn`` folds the qkv weight (see the ``fused_qkv_rope_ln`` docstring
for why the fold cannot be done inside the k loop).  ``Wd`` ``[H, I]`` is *not*
folded: ``H`` is its output axis, not its contraction axis.

``rstd`` is per **row**, so it cannot fold into a weight and cannot be factored
out of silu (``silu(rstd * g) != rstd * silu(g)``).  It is applied to the fp32
``g_acc``/``u_acc`` accumulators after the k sweep and before the epilogue, so
the projection is the model's, with the one rounding difference the qkv fold
already documents: the model normalizes a bf16-rounded activation, this
normalizes the fp32 accumulator.

The residual add stays on the driver: the kernel takes the raw ``hidden +
residual`` sum, which is also the value the layer carries forward as its next
residual, so the kernel never writes it back.

Why the variance pass is shaped the way it is
---------------------------------------------
It borrows three tiles that are dead until the silu epilogue -- ``mid`` (the bf16
DMA landing tile), ``sig`` (fp32 widen+square) and ``mid_f`` (fp32 chunk sum) --
and adds only the three ``[block_M]`` fp32 vectors ``ss``/``var``/``rstd``.  A
dedicated ``[block_M, block_K]`` fp32 scratch would be 64 KB/lane at
``block_M=1024``, which takes the kernel's ~195 KB past the 256 KB per-lane
budget.  The reuse is legal because every tile is declared at kernel entry and
nothing in the emitter tracks liveness -- but it is a *human* invariant: if the I
sweep is ever converted to ``T.Pipelined``, ``lower_tpu_pipeline`` may clone the
buffers written in the prefetch stage and the borrowing breaks.

The pass re-reads ``X`` through a global ``T.copy`` instead of slicing the
resident ``x_t``.  It has to: the emitter resolves a ``BufferLoad`` of a local
tile to the *whole tile name* and ignores the indices (``_emit_into`` /
``_gen_value``), so ``x_t[i, k0 + j]`` would silently emit a whole-tile cast of
the wrong width.  The extra read is one ``[T, H]`` bf16 pass, against the ~150 MB
of weight stream the row tiles already move.

Squaring into ``d_acc`` -- it is ``[block_M, H]`` fp32 and dead before
``T.fill(d_acc, 0)`` -- would make the pass free, but it needs ``block_K == H``
and a single ``H``-wide reduce, and no kernel in this tree reduces more than
``block_I`` (256) columns.  Left as a follow-up behind a hardware probe.

Everything else -- the grid, the row tiles, the ``block_K`` hoist of ``x_t``,
the silu epilogue's fp32 discipline -- is ``fused_mlp``'s, unchanged, and its
module docstring is the place to read about block sizes and local memory.

Contract
--------
``TP`` (token rows) is padded up to a multiple of ``block_M`` by the driver, as
in ``fused_mlp``; padded rows read garbage, normalize garbage and are dropped by
the driver's slice.

``X`` is the *un-normalized* ``hidden + residual``.  ``Wgu`` is **already folded
with the post-attention gamma**, i.e. ``gamma * gate_up_proj.weight`` (``[2I, H]``,
gate rows first); ``Wd`` is ``down_proj.weight`` as stored.  ``ln_eps`` is baked;
both the RMSNorm epsilon and the activation dtype are part of the build.
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
def fused_mlp_ln(
    hidden: int = 1024,
    intermediate: int = 3072,
    block_M: int = 128,
    block_I: int = 128,
    block_K: int = 256,
    ln_eps: float = 1e-6,
    dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
):
    """Fused post-attention norm + gate/up projection + silu_mul + down.

    ``block_M``/``block_I``/``block_K`` mean what they mean in ``fused_mlp``;
    the variance pass additionally walks the row in ``block_I``-wide chunks, so
    ``hidden`` must be a multiple of it.
    """
    H = int(hidden)
    I = int(intermediate)
    assert I % block_I == 0, "intermediate must be a multiple of block_I"
    assert H % block_K == 0, "hidden must be a multiple of block_K"
    assert H % block_I == 0, "hidden must tile into block_I-wide norm chunks"
    # Number of k steps per (gate, up) pair.  When it is one -- the shipped
    # configuration, since ``block_K`` is chosen as large as ``hidden`` allows --
    # the X tile does not change between I blocks, so it is loaded once for the
    # whole sweep instead of once per block.
    k_steps = H // block_K
    inv_h = 1.0 / H
    ln_eps_f = float(ln_eps)

    @T.prim_func
    def main(
        X: T.Tensor((TP, H), dtype),
        Wgu: T.Tensor((2 * I, H), dtype),
        Wd: T.Tensor((H, I), dtype),
        Y: T.Tensor((TP, H), dtype),
    ):
        Tp, _ = X.shape
        Mt = T.ceildiv(Tp, block_M)
        tiles_per_core = T.ceildiv(Mt, _NUM_CORES)

        with T.Kernel(Mt, 1, _NUM_CORES) as (bx, by, bc):
            x_t = T.alloc_local((block_M, block_K), dtype)
            w_g = T.alloc_local((block_I, block_K), dtype)
            w_u = T.alloc_local((block_I, block_K), dtype)
            w_d = T.alloc_local((H, block_I), dtype)
            g_acc = T.alloc_local((block_M, block_I), accum_dtype)
            u_acc = T.alloc_local((block_M, block_I), accum_dtype)
            # sig holds 1 + exp(gate) -- the silu denominator.  fp32 on purpose:
            # the emitter materializes nested subexpressions in the *store's*
            # dtype, so anything narrower would run fdiv/fmul in that dtype.
            # Before the I sweep it is the norm pass's widening scratch.
            sig = T.alloc_local((block_M, block_I), accum_dtype)
            mid_f = T.alloc_local((block_M, block_I), accum_dtype)
            mid = T.alloc_local((block_M, block_I), dtype)
            d_acc = T.alloc_local((block_M, H), accum_dtype)
            # The input norm's per-row statistics.  Only these three are new
            # relative to ``fused_mlp``: the pass borrows ``mid``/``sig``/``mid_f``
            # (see the module docstring) to stay inside the local budget.
            ss = T.alloc_local((block_M,), accum_dtype)
            var = T.alloc_local((block_M,), accum_dtype)
            in_rstd = T.alloc_local((block_M,), accum_dtype)

            g = bc * tiles_per_core + bx
            with T.If(g < Mt), T.Then():
                row0 = g * block_M
                T.fill(d_acc, 0)
                if k_steps == 1:
                    # Loop-invariant: fetch the activation tile once and hold it
                    # across the whole I sweep.  Leaving the copy in the k loop
                    # below re-reads it I/block_I times -- 12x at the shipped
                    # tiles, ~190 MB per layer at T=8192 -- and buys nothing.
                    T.copy(X[row0, 0], x_t)

                # --- input RMSNorm scale, once per row tile ------------------ #
                # Walk the row once for sum(x^2), in block_I-wide chunks so the
                # reduce stays the width the rest of the tree runs.  The squares
                # accumulate in a tile and are reduced once at the end: a
                # ``clear=False`` reduce *inside* the loop makes the emitter
                # allocate a fresh reduce temp per iteration, which the local
                # allocator does not survive.
                T.fill(mid_f, 0)
                for kk in T.serial(H // block_I):
                    T.copy(X[row0, kk * block_I], mid)
                    # PPL has no implicit widening: a bf16 square cannot enter
                    # ``quick_pooling`` (source and destination must share one
                    # DataType) and would lose bf16 precision anyway.
                    for i, j in T.Parallel(block_M, block_I):
                        sig[i, j] = T.Cast("float32", mid[i, j])
                    for i, j in T.Parallel(block_M, block_I):
                        sig[i, j] = sig[i, j] * sig[i, j]
                    for i, j in T.Parallel(block_M, block_I):
                        mid_f[i, j] = mid_f[i, j] + sig[i, j]
                T.reduce_sum(mid_f, ss, dim=1)
                for i in T.Parallel(block_M):
                    var[i] = ss[i] * inv_h
                for i in T.Parallel(block_M):
                    var[i] = var[i] + ln_eps_f
                for i in T.Parallel(block_M):
                    in_rstd[i] = T.rsqrt(var[i])

                for ib in T.serial(I // block_I):
                    i0 = ib * block_I
                    T.fill(g_acc, 0)
                    T.fill(u_acc, 0)
                    # Gate and up share the X tile, so one load per k-step.
                    for kb in T.serial(k_steps):
                        k0 = kb * block_K
                        if k_steps > 1:
                            T.copy(X[row0, k0], x_t)
                        T.copy(Wgu[i0, k0], w_g)
                        gemm_tiled(x_t, w_g, g_acc, transpose_B=True)
                        T.copy(Wgu[I + i0, k0], w_u)
                        gemm_tiled(x_t, w_u, u_acc, transpose_B=True)

                    # The norm's per-row scale, on the fp32 accumulators and
                    # before the epilogue.  Two passes, not a nested expression:
                    # ``T.exp(g_acc[i, j] * in_rstd[i])`` would materialize the
                    # product into a fresh temp tile (``_gen_value``), which is
                    # the same reason the silu chain below is five stores.
                    for i, j in T.Parallel(block_M, block_I):
                        g_acc[i, j] = g_acc[i, j] * in_rstd[i]
                    for i, j in T.Parallel(block_M, block_I):
                        u_acc[i, j] = u_acc[i, j] * in_rstd[i]

                    # silu(gate) * up, on the fp32 accumulators.
                    #
                    # The gates use x*exp(x)/(1+exp(x)) rather than
                    # x/(1+exp(-x)): it is the same sigmoid (the SDK's own
                    # ``sigmoid_fp32``) without the negation pass.
                    # ``exp_no_overflow`` saturates at +-3.4e35, so a large |x|
                    # degrades to exactly x or 0 -- never inf*0.
                    #
                    # Order matters: the numerator takes exp(gate) before
                    # ``sig`` is overwritten with the denominator.  Each store
                    # is a whole-tile pass, and the emitter can only express
                    # ``T.exp`` as a standalone statement.
                    for i, j in T.Parallel(block_M, block_I):
                        sig[i, j] = T.exp(g_acc[i, j])
                    for i, j in T.Parallel(block_M, block_I):
                        mid_f[i, j] = g_acc[i, j] * sig[i, j]
                    for i, j in T.Parallel(block_M, block_I):
                        sig[i, j] = 1.0 + sig[i, j]
                    for i, j in T.Parallel(block_M, block_I):
                        mid_f[i, j] = mid_f[i, j] / sig[i, j]
                    for i, j in T.Parallel(block_M, block_I):
                        mid_f[i, j] = mid_f[i, j] * u_acc[i, j]
                    # The one narrowing: the down GEMM takes bf16 operands, and
                    # this is the same rounding point the torch chain has.
                    T.copy(mid_f, mid)

                    T.copy(Wd[0, i0], w_d)
                    gemm_tiled(mid, w_d, d_acc, transpose_B=True)

                T.copy(d_acc, Y[row0, 0])

    return main
