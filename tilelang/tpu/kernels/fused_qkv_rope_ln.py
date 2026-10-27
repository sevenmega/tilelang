"""Fused input-RMSNorm -> qkv projection -> qk-RMSNorm -> RoPE -> KV-cache write.

Milestone B of ``fused_qkv_rope``: the same kernel with the decoder layer's
``input_layernorm`` folded in, so the whole pre-attention chain is one launch::

    residual = hidden + residual             # the layer does this, see below
    normed   = input_layernorm(residual)     # <- this kernel
    qkv      = self.qkv_proj(normed)         # <- this kernel
    q, k     = apply_qk_norm(q, k, ...)      # <- this kernel
    q, k     = self.rotary_emb(...)          # <- this kernel
    k_cache[loc], v_cache[loc] = k, v       # <- this kernel

Two things make it fit in the same tile budget as the projection-only kernel.

**The input-norm variance needs the whole row, the GEMM streams it.**  With
``block_M`` rows on chip and a ``block_K``-wide k-tile, the row norm can only be
finished before the first ``fmm2`` if the k axis is walked twice: once for
``sum(x^2)``, once for the projection.  That is what this kernel does, and the
second pass is free -- the projection loop was already re-reading ``X`` per
head.  The sum cannot ride on the accumulator because ``tiu::fadd``/``fmul``
demand one DataType across all three tiles, so a bf16 tile only enters the fp32
reduction through an explicit ``T.Cast``, which the emitter turns into a
``tiu::cast`` widening (exact).  ``ppl_tiu_func.h`` also declares a
``fmul_cast(dst, src, 1.0f)`` spelling of the same thing, but ppl-compile
rejects it here as "unsupported arith type".

**The input gamma folds into the projection weight, exactly.**  ``x_norm = (x *
rstd) * gamma_in`` and ``q = x_norm . W^T``, so ``q = rstd * (x . (gamma_in *
W)^T)``: ``gamma_in`` indexes the contraction axis, so it is a per-**column**
scale of ``W`` -- unlike ``qk_norm``'s gamma, which is followed by RoPE and
cannot be folded.  ``W`` is a parameter, so the driver folds it **once per
layer** and this kernel takes the pre-folded weight ``gamma_in * W_qkv``; the
k-loop is then byte for byte the projection-only kernel's.

Folding the slab *inside* the k-loop was tried and is wrong: an elementwise TIU
op between a weight slab's ``dma::load`` and the ``fmm2`` that consumes it
corrupts the accumulator (measured on hardware -- q/k off by 13 % with a random
``gamma_in`` and by 64 % with ``gamma_in = 1``, where the multiply is an exact
no-op, while the same GEMM on a pre-folded weight matches to bf16 precision).
The driver-side fold is one ``[Ntot, H]`` elementwise pass per layer, i.e. no
per-launch work at all, so nothing is lost.

**The input rstd is only needed where nothing divides it out.**  It is a
positive per-row scalar, so ``rms(rstd * p) = rstd * rms(p)``: it cancels
exactly in the qk-norm (``n = p / rms(p) * gamma``) and in RoPE, which is
linear.  ``q`` and ``k`` therefore need no scaling at all, and ``v`` -- which
has no norm after it -- is scaled by ``in_rstd`` on the fp32 accumulator.

The residual add itself stays on the driver (`Qwen3DecoderLayer.forward`): the
kernel takes the *un-normalized* sum, which is also the value the layer carries
forward as its next residual, so the kernel never writes it back.
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
def fused_qkv_rope_ln(
    hidden: int = 1024,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    block_M: int = 64,
    block_K: int = 256,
    eps: float = 1e-6,
    ln_eps: float = 1e-6,
    dtype: T.dtype = T.bfloat16,
    accum_dtype: T.dtype = T.float32,
):
    """Fused input-RMSNorm + projection + qk-norm + RoPE + paged-cache write.

    ``eps`` is the qk-norm epsilon, ``ln_eps`` the input-layernorm one; both are
    baked.  ``X`` is the *un-normalized* hidden (``hidden + residual``) and ``W``
    the qkv weight **already folded with the input gamma**, i.e.
    ``gamma_in * W_qkv`` (see the module docstring).
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
    ln_eps_f = float(ln_eps)
    inv_h = 1.0 / H

    # The row-norm pass walks the row in ``half``-wide chunks so its fp32 widen
    # scratch is the ``n_hi`` tile the qk-norm already owns; a ``block_K`` (256)
    # chunk would need a 64 KB fp32 tile of its own.
    assert H % half == 0, "hidden must be a multiple of head_dim/2"

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
            # bf16 staging for the input-norm pass -- the fp32 widen lands in
            # ``n_hi``, so this costs one bf16 load tile.
            xs = T.alloc_local((block_M, half), dtype)
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
            in_ss = T.alloc_local((block_M,), accum_dtype)
            in_var = T.alloc_local((block_M,), accum_dtype)
            in_rstd = T.alloc_local((block_M,), accum_dtype)

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

                    # --- input RMSNorm scale, once per m-tile ------------- #
                    # Sits between ``k`` and ``v`` on purpose: ``v`` is its only
                    # consumer (the input ``rstd`` cancels in q/k's own norm and
                    # RoPE), so the q/k heads above stay byte-identical to the
                    # projection-only kernel.
                    #
                    # Walk the row once for sum(x^2); the projection loops walk
                    # it again per head anyway, so nothing extra is read.  The
                    # squares are accumulated in a tile and reduced once at the
                    # end: a ``clear=False`` reduce *inside* the loop makes the
                    # emitter allocate a fresh reduce temp per iteration
                    # (``make_tensor`` in the loop body), and the local-memory
                    # allocator does not survive that.
                    T.fill(n_lo, 0)
                    for kk in T.serial(H // half):
                        T.copy(X[row0, kk * half], xs)
                        for i, j in T.Parallel(block_M, half):
                            n_hi[i, j] = T.Cast("float32", xs[i, j])
                        for i, j in T.Parallel(block_M, half):
                            n_hi[i, j] = n_hi[i, j] * n_hi[i, j]
                        for i, j in T.Parallel(block_M, half):
                            n_lo[i, j] = n_lo[i, j] + n_hi[i, j]
                    T.reduce_sum(n_lo, in_ss, dim=1)
                    for i in T.Parallel(block_M):
                        in_var[i] = in_ss[i] * inv_h
                    for i in T.Parallel(block_M):
                        in_var[i] = in_var[i] + ln_eps_f
                    for i in T.Parallel(block_M):
                        in_rstd[i] = T.rsqrt(in_var[i])

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

                    # ``v`` is the normalized hidden times W_v, so it carries the
                    # input norm's per-row scale (and none of the qk-norm).
                    for i, j in T.Parallel(block_M, half):
                        acc_lo[i, j] = acc_lo[i, j] * in_rstd[i]
                    T.copy(acc_lo, out_lo)
                    T.copy(KVRow[2 * g, row0], kv_idx)
                    T.call_extern(
                        "handle", "tl_tpu_scatter_rows", Vc.data, kv_idx.data,
                        out_lo.data,
                    )
                    for i, j in T.Parallel(block_M, half):
                        acc_hi[i, j] = acc_hi[i, j] * in_rstd[i]
                    T.copy(acc_hi, out_hi)
                    T.copy(KVRow[2 * g + 1, row0], kv_idx)
                    T.call_extern(
                        "handle", "tl_tpu_scatter_rows", Vc.data, kv_idx.data,
                        out_hi.data,
                    )

    return main
