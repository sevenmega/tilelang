"""Fused qkv projection -> qk-RMSNorm -> RoPE -> KV-cache write, for TPU.

One launch replaces the six-stage pre-attention chain every Qwen3 decoder layer
runs as plain torch today::

    qkv, _ = self.qkv_proj(hidden)          # [T, (Hq + 2*Hkv) * D]
    q, k, v = qkv.split(...)                # strided views
    q, k = apply_qk_norm(q, k, ...)         # reshape(-1, D), 2 RMSNorms
    q, k = self.rotary_emb(positions, q, k) # index_select + chunk/mul/add/cat
    k_cache[out_cache_loc] = k              # 2 strided scatters
    v_cache[out_cache_loc] = v

The kernel writes ``Q`` for the paged attention kernel and writes ``K`` / ``V``
straight into the paged KV cache, so the driver never materializes ``k`` / ``v``.

Layout notes
------------
``KV`` pairs lanes ``(d, d + D/2)``, and neither a per-element re-index inside a
tile nor a reassembly of two half tiles is expressible in the TPU DSL.  Each
head is therefore accumulated as two ``D/2``-wide halves -- two ``fmm2`` calls
instead of one, same FLOPs -- and RoPE is whole-tile arithmetic::

    out_lo = n_lo * cos - n_hi * sin
    out_hi = n_hi * cos + n_lo * sin

``n = x * rstd * gamma`` is applied to the **fp32 accumulator**, before the
rotation: the norms are not foldable into the projection weight (``(g x) /
rms(g x) != g x / rms(x)``, and RoPE mixes lanes with different ``gamma``), and
normalizing the fp32 accumulation is one rounding *tighter* than the reference,
which normalizes the bf16-rounded projection.

The paged cache is ``[Slots, Hkv, D]`` bf16, i.e. flat rows of ``D/2`` elements
``slot * 2 * Hkv + 2 * h + half``.  ``Kc`` / ``Vc`` are handed in under that
``[Slots * 2 * Hkv, D/2]`` view and each half is written with
``tl_tpu_scatter_rows`` -- the write-side mirror of the attention kernel's
``tl_tpu_gather_rows`` -- by the row index table ``KVRow[h2, t]``.

Jobs / grid
-----------
The last grid axis is the core (``ppl_codegen`` lowers a 3-axis ``T.Kernel`` to
``get_block_index()`` plus a per-core slice).  One grid job owns one *KV head*
and everything that depends on it: the ``ratio = Hq // Hkv`` query heads
sharing it, then its own ``k`` and ``v`` head.  That keeps every job identical
(no divergence between cores) and every q/k/v head written exactly once, and the
``ratio + 2`` heads of weight it streams stay in L2 across the m-tile loop.

The m-tile loop is *inside* the job, so a job's weight slab is read once from
DRAM per launch instead of once per m-tile.

Contract
--------
``T`` (token rows) is padded up to a multiple of ``block_M`` -- tile extents are
baked and cannot be clamped.  Padded rows read garbage from ``X`` / ``CosRows``
(harmless: their ``Q`` rows are sliced off) and their ``KVRow`` entries point at
the pool's reserved pad slot, never at a live slot.

``X`` is the *normalized* hidden state (``input_layernorm`` stays where it is);
folding it in is the next milestone.
"""

import tilelang
import tilelang.language as T

from tilelang.tpu.op.gemm.tiled import gemm_tiled

# Dynamic extents.  ``TP`` is the token count padded to ``block_M``; ``S2`` is
# the paged cache's flat row count, ``Slots * 2 * Hkv``.  Everything else --
# the head geometry, ``H``, the tile sizes -- is baked, so one build serves one
# model and every batch shape.
TP = T.dynamic("TP", "int32")
S2 = T.dynamic("S2", "int32")

# SG2260E has four cores; the last grid axis is the core index. Baked.
_NUM_CORES = 4


@tilelang.jit(target="tpu")
def fused_qkv_rope(
    hidden: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_M: int = 64,
    block_K: int = 256,
    eps: float = 1e-6,
    dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
):
    """Fused projection + qk-norm + RoPE + paged-cache write.

    ``hidden``/``num_heads``/``num_kv_heads``/``head_dim`` describe the model
    (Qwen3-0.6B defaults); ``block_M`` is the query-row tile, ``block_K`` the
    k-reduction tile.  ``ratio = num_heads // num_kv_heads`` must divide evenly.
    """
    H = int(hidden)
    Hq = int(num_heads)
    Hkv = int(num_kv_heads)
    D = int(head_dim)
    assert Hq % Hkv == 0, "GQA ratio must be integral"
    ratio = Hq // Hkv
    assert D % 2 == 0, "head_dim must be even for half-split RoPE"
    half = D // 2

    Ntot = (Hq + 2 * Hkv) * D        # q | k | v rows of the fused weight
    Nq = Hq * D                     # [TP, Nq] query output
    inv_d = 1.0 / D
    eps_f = float(eps)

    # Rows of each head kind in the fused weight (rows [q | k | v]).
    def _q_row(h):
        return h * D

    def _k_row(h):
        return Hq * D + h * D

    def _v_row(h):
        return (Hq + Hkv) * D + h * D

    groups_per_core = (Hkv + _NUM_CORES - 1) // _NUM_CORES

    @T.prim_func
    def main(
        X: T.Tensor((TP, H), dtype),
        W: T.Tensor((Ntot, H), dtype),
        CosRows: T.Tensor((TP, D), T.float32),
        Gamma: T.Tensor((2, D), T.float32),
        KVRow: T.Tensor((2 * Hkv, TP), T.uint32),
        Kc: T.Tensor((S2, half), dtype),
        Vc: T.Tensor((S2, half), dtype),
        QOut: T.Tensor((TP, Nq), dtype),
    ):
        with T.Kernel(Hkv, 1, _NUM_CORES) as (bx, by, bc):
            x_tile = T.alloc_local((block_M, block_K), dtype)
            w_lo = T.alloc_local((half, block_K), dtype)
            w_hi = T.alloc_local((half, block_K), dtype)
            acc_lo = T.alloc_local((block_M, half), accum_dtype)
            acc_hi = T.alloc_local((block_M, half), accum_dtype)
            n_lo = T.alloc_local((block_M, half), accum_dtype)
            n_hi = T.alloc_local((block_M, half), accum_dtype)
            # cos / sin are fp32 on purpose: bf16->fp32 `tiu::cast` does not
            # exist, and the elementwise ops cannot mix a bf16 tile with the
            # fp32 accumulator.
            cos_t = T.alloc_local((block_M, half), T.float32)
            sin_t = T.alloc_local((block_M, half), T.float32)
            sq = T.alloc_local((block_M, half), accum_dtype)
            out_lo = T.alloc_local((block_M, half), dtype)
            out_hi = T.alloc_local((block_M, half), dtype)
            gq_lo = T.alloc_local((1, half), T.float32)
            gq_hi = T.alloc_local((1, half), T.float32)
            gk_lo = T.alloc_local((1, half), T.float32)
            gk_hi = T.alloc_local((1, half), T.float32)
            kv_idx = T.alloc_local((1, block_M), T.uint32)
            ss = T.alloc_local((block_M,), accum_dtype)
            var = T.alloc_local((block_M,), accum_dtype)
            rstd = T.alloc_local((block_M,), accum_dtype)

            # qk-norm weights are per *lane* and shared by every head of their
            # kind, so they are loaded once per core.
            T.copy(Gamma[0, 0], gq_lo)
            T.copy(Gamma[0, half], gq_hi)
            T.copy(Gamma[1, 0], gk_lo)
            T.copy(Gamma[1, half], gk_hi)

            g = bc * groups_per_core + bx
            with T.If(g < Hkv), T.Then():
                for mt in T.serial(TP // block_M):
                    row0 = mt * block_M
                    T.copy(CosRows[row0, 0], cos_t)
                    T.copy(CosRows[row0, half], sin_t)

                    # --- query heads of this KV head ---------------------- #
                    for s in T.serial(ratio):
                        qh = g * ratio + s
                        T.fill(acc_lo, 0)
                        T.fill(acc_hi, 0)
                        for kk in T.serial(H // block_K):
                            k0 = kk * block_K
                            T.copy(X[row0, k0], x_tile)
                            T.copy(W[_q_row(qh), k0], w_lo)
                            gemm_tiled(x_tile, w_lo, acc_lo, transpose_B=True)
                            T.copy(W[_q_row(qh) + half, k0], w_hi)
                            gemm_tiled(x_tile, w_hi, acc_hi, transpose_B=True)

                        # rstd from the full D-wide head, then n = x*rstd*gamma
                        for i, j in T.Parallel(block_M, half):
                            sq[i, j] = acc_lo[i, j] * acc_lo[i, j]
                        T.reduce_sum(sq, ss, dim=1)
                        for i, j in T.Parallel(block_M, half):
                            sq[i, j] = acc_hi[i, j] * acc_hi[i, j]
                        T.reduce_sum(sq, ss, dim=1, clear=False)
                        for i in T.Parallel(block_M):
                            var[i] = ss[i] * inv_d
                        for i in T.Parallel(block_M):
                            var[i] = var[i] + eps_f
                        for i in T.Parallel(block_M):
                            rstd[i] = T.rsqrt(var[i])
                        for i, j in T.Parallel(block_M, half):
                            n_lo[i, j] = acc_lo[i, j] * rstd[i]
                        for i, j in T.Parallel(block_M, half):
                            n_lo[i, j] = n_lo[i, j] * gq_lo[0, j]
                        for i, j in T.Parallel(block_M, half):
                            n_hi[i, j] = acc_hi[i, j] * rstd[i]
                        for i, j in T.Parallel(block_M, half):
                            n_hi[i, j] = n_hi[i, j] * gq_hi[0, j]

                        # RoPE.  ``acc_lo`` / ``acc_hi`` are free scratch now
                        # (their content is in n_*), and they land in a bf16
                        # tile only through ``T.copy`` -- a bf16 store of an
                        # fp32 expression needs an explicit Cast.
                        for i, j in T.Parallel(block_M, half):
                            acc_lo[i, j] = n_lo[i, j] * cos_t[i, j]
                        for i, j in T.Parallel(block_M, half):
                            n_lo[i, j] = n_lo[i, j] * sin_t[i, j]
                        for i, j in T.Parallel(block_M, half):
                            acc_hi[i, j] = n_hi[i, j] * cos_t[i, j]
                        for i, j in T.Parallel(block_M, half):
                            n_hi[i, j] = n_hi[i, j] * sin_t[i, j]
                        for i, j in T.Parallel(block_M, half):
                            acc_lo[i, j] = acc_lo[i, j] - n_hi[i, j]
                        for i, j in T.Parallel(block_M, half):
                            acc_hi[i, j] = acc_hi[i, j] + n_lo[i, j]

                        T.copy(acc_lo, out_lo)
                        T.copy(acc_hi, out_hi)
                        T.copy(out_lo, QOut[row0, qh * D])
                        T.copy(out_hi, QOut[row0, qh * D + half])

                    # --- key head ----------------------------------------- #
                    T.fill(acc_lo, 0)
                    T.fill(acc_hi, 0)
                    for kk in T.serial(H // block_K):
                        k0 = kk * block_K
                        T.copy(X[row0, k0], x_tile)
                        T.copy(W[_k_row(g), k0], w_lo)
                        gemm_tiled(x_tile, w_lo, acc_lo, transpose_B=True)
                        T.copy(W[_k_row(g) + half, k0], w_hi)
                        gemm_tiled(x_tile, w_hi, acc_hi, transpose_B=True)

                    for i, j in T.Parallel(block_M, half):
                        sq[i, j] = acc_lo[i, j] * acc_lo[i, j]
                    T.reduce_sum(sq, ss, dim=1)
                    for i, j in T.Parallel(block_M, half):
                        sq[i, j] = acc_hi[i, j] * acc_hi[i, j]
                    T.reduce_sum(sq, ss, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        var[i] = ss[i] * inv_d
                    for i in T.Parallel(block_M):
                        var[i] = var[i] + eps_f
                    for i in T.Parallel(block_M):
                        rstd[i] = T.rsqrt(var[i])
                    for i, j in T.Parallel(block_M, half):
                        n_lo[i, j] = acc_lo[i, j] * rstd[i]
                    for i, j in T.Parallel(block_M, half):
                        n_lo[i, j] = n_lo[i, j] * gk_lo[0, j]
                    for i, j in T.Parallel(block_M, half):
                        n_hi[i, j] = acc_hi[i, j] * rstd[i]
                    for i, j in T.Parallel(block_M, half):
                        n_hi[i, j] = n_hi[i, j] * gk_hi[0, j]

                    for i, j in T.Parallel(block_M, half):
                        acc_lo[i, j] = n_lo[i, j] * cos_t[i, j]
                    for i, j in T.Parallel(block_M, half):
                        n_lo[i, j] = n_lo[i, j] * sin_t[i, j]
                    for i, j in T.Parallel(block_M, half):
                        acc_hi[i, j] = n_hi[i, j] * cos_t[i, j]
                    for i, j in T.Parallel(block_M, half):
                        n_hi[i, j] = n_hi[i, j] * sin_t[i, j]
                    for i, j in T.Parallel(block_M, half):
                        acc_lo[i, j] = acc_lo[i, j] - n_hi[i, j]
                    for i, j in T.Parallel(block_M, half):
                        acc_hi[i, j] = acc_hi[i, j] + n_lo[i, j]

                    T.copy(acc_lo, out_lo)
                    T.copy(acc_hi, out_hi)

                    # Cache rows of this head: ``slot * 2 * Hkv + 2 * g + half``.
                    T.copy(KVRow[2 * g, row0], kv_idx)
                    T.call_extern(
                        "handle", "tl_tpu_scatter_rows", Kc.data, kv_idx.data,
                        out_lo.data,
                    )
                    T.copy(KVRow[2 * g + 1, row0], kv_idx)
                    T.call_extern(
                        "handle", "tl_tpu_scatter_rows", Kc.data, kv_idx.data,
                        out_hi.data,
                    )

                    # --- value head --------------------------------------- #
                    T.fill(acc_lo, 0)
                    T.fill(acc_hi, 0)
                    for kk in T.serial(H // block_K):
                        k0 = kk * block_K
                        T.copy(X[row0, k0], x_tile)
                        T.copy(W[_v_row(g), k0], w_lo)
                        gemm_tiled(x_tile, w_lo, acc_lo, transpose_B=True)
                        T.copy(W[_v_row(g) + half, k0], w_hi)
                        gemm_tiled(x_tile, w_hi, acc_hi, transpose_B=True)

                    T.copy(acc_lo, out_lo)
                    T.copy(KVRow[2 * g, row0], kv_idx)
                    T.call_extern(
                        "handle", "tl_tpu_scatter_rows", Vc.data, kv_idx.data,
                        out_lo.data,
                    )
                    T.copy(acc_hi, out_hi)
                    T.copy(KVRow[2 * g + 1, row0], kv_idx)
                    T.call_extern(
                        "handle", "tl_tpu_scatter_rows", Vc.data, kv_idx.data,
                        out_hi.data,
                    )

    return main
