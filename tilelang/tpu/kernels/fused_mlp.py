"""Fused ``gate_up_proj -> silu_mul -> down_proj`` as a tilelang DSL kernel for TPU.

The torch chain it replaces is three launches and a 100 MB bf16 intermediate::

    gate_up, _ = gate_up_proj(x)          # [T, 2I]  -> DRAM
    mid        = silu(gate_up[..., :I]) * gate_up[..., I:]
    y, _       = down_proj(mid)           # [T, H]

Both the write of ``[T, 2I]`` and the two reads of ``[T, I]`` are DDR traffic --
at T=8192 they are ~200 MB of the MLP tail's ~330 MB, and the two elementwise
rows of the profile are bound by them, not by math.  This kernel computes the
gate/up projections, applies silu and multiplies in registers, and feeds the
product straight into the down projection, so the intermediate never leaves the
core.

Why the shape is what it is
---------------------------
``down_proj`` needs the **whole** I reduction before it can write a row, so a job
owns complete rows and keeps ``[block_M, H]`` fp32 of down accumulator in local
memory across the entire I sweep.  That accumulator is why ``block_M`` -- not
``block_I`` -- is the knob that decides cost: a row tile streams the full weight
set (gate 6.3 + up 6.3 + down 6.3 MB at Qwen3-0.6B), so DRAM traffic is
``(row tiles per core) * 18.9 MB``.  Four cores walk the same weight sequence in
lockstep on different rows, so the concurrent fetches share one stream.

The activation tile is the other end of it.  ``block_K`` is chosen as large as
``hidden`` allows, so the k loop collapses to a single step and the X tile is
**loop-invariant across the I sweep**: it is loaded once per row tile, not once
per (I block, k step).  Left inside the I loop -- as it was when the kernel was
first tuned -- it is re-read ``I / block_I`` times, 12x at the shipped tiles,
which is ~190 MB of redundant traffic per layer at T=8192.  Hoisting is free:
``x_t`` is already a full-row tile, so its local-memory cost is the same either
way.  It took T=8192 from 4.41 ms to 3.92 ms (1.70x -> 1.87x vs the torch
chain).

When ``hidden`` is not a multiple of the largest ``block_K`` the driver drops to
a smaller one, ``k_steps > 1``, and the copy stays in the k loop -- there the
tile genuinely changes per step.

TPU local memory is accounted per *lane* (256 KB each; a ``{1, r, 1, c}`` tile
costs ``ceil(r/64) * align(c, 16) * dsize`` bytes), so a ``[512, 1024]`` fp32
accumulator is 32 KB/lane, not 512 KB.  Large row tiles are affordable, which is
what makes the weight stream small enough to sit under the GEMM time.

Contract
--------
``TP`` (token rows) is padded up to a multiple of ``block_M`` by the driver --
tile extents are baked and cannot be clamped.  Padded rows read garbage from
``X`` and produce output the driver drops by slicing ``y[:num_tokens]``.

``X`` is the *already normalized* hidden state: ``post_attention_layernorm`` runs
before the MLP, so nothing here needs a norm.  Weights are consumed exactly as
stored (``gate_up_proj.weight`` is ``[2I, H]`` with gate first, ``down_proj`` is
``[H, I]``), through ``transpose_B=True`` -- no driver-side prepacking.
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
def fused_mlp(
    hidden: int = 1024,
    intermediate: int = 3072,
    block_M: int = 128,
    block_I: int = 128,
    block_K: int = 256,
    dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
):
    """Fused gate/up projection + silu_mul + down projection.

    ``block_M`` is the row tile (the dominant knob: see the module docstring);
    ``block_I`` the intermediate-axis tile (gate/up accumulator width and the
    down weight slab width); ``block_K`` the reduction tile over ``hidden``.
    """
    H = int(hidden)
    I = int(intermediate)
    assert I % block_I == 0, "intermediate must be a multiple of block_I"
    assert H % block_K == 0, "hidden must be a multiple of block_K"
    # Number of k steps per (gate, up) pair.  When it is one -- the shipped
    # configuration, since ``block_K`` is chosen as large as ``hidden`` allows --
    # the X tile does not change between I blocks, so it is loaded once for the
    # whole sweep instead of once per block.
    k_steps = H // block_K

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
            # The whole down weight slab for one I block: [H, block_I].  It is
            # only 4 KB/lane per 128 columns, and keeping H whole is what lets
            # the accumulator be a single tile.
            w_d = T.alloc_local((H, block_I), dtype)
            g_acc = T.alloc_local((block_M, block_I), accum_dtype)
            u_acc = T.alloc_local((block_M, block_I), accum_dtype)
            # sig holds 1 + exp(gate) -- the silu denominator.  fp32 on purpose:
            # the emitter materializes nested subexpressions in the *store's*
            # dtype, so anything narrower would run fdiv/fmul in that dtype.
            sig = T.alloc_local((block_M, block_I), accum_dtype)
            mid_f = T.alloc_local((block_M, block_I), accum_dtype)
            mid = T.alloc_local((block_M, block_I), dtype)
            # Live across the whole I sweep: that is the fusion's cost, and the
            # reason block_M is chosen, not maximised.
            d_acc = T.alloc_local((block_M, H), accum_dtype)

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
