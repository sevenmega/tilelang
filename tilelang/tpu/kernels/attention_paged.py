"""Paged/varlen flash-attention as a tilelang DSL kernel for TPU.

Companion to :mod:`tilelang.tpu.kernels.attention` (the one-request-per-call
``flash_attention_gqa``).  This variant reads a paged KV cache **in place** and
takes the whole batch's queries packed, so the sglang driver does no per-request
gather / pack / scatter.

It lives in its own module on purpose: its dynamic-shape variables are declared
here, independent of the legacy kernel's, so the two never share a ``T.dynamic``
symbol or a kernel identity.  Keeping them apart is what lets both
``test_flash_attention.py`` and ``test_flash_attention_paged.py`` build and run
without interfering (the same reason ``flash_attention_v1_static`` is separate).
"""

import tilelang
import tilelang.language as T

from tilelang.tpu.op.gemm.tiled import gemm_tiled

# Dynamic shapes for the paged kernel -- declared here, NOT imported from the
# legacy module, so nothing is shared.  ``Tk`` is the packed token count of the
# whole batch, ``Slots`` the KV-cache row count, ``Bk``/``QP``/``KV`` the request
# count and the bucket's padded q / kv extents.  ``D`` (head dim) stays baked.
Tk = T.dynamic("Tk", "int32")
Slots = T.dynamic("Slots", "int32")
Hq = T.dynamic("Hq", "int32")
Hkv = T.dynamic("Hkv", "int32")
Bk = T.dynamic("Bk", "int32")
QP = T.dynamic("QP", "int32")
KV = T.dynamic("KV", "int32")

# SG2260E has four cores; the last grid axis is the core index. Baked.
_NUM_CORES = 4


@tilelang.jit(target="tpu")
def flash_attention_paged(
    d: int = 128,
    block_M: int = 64,
    block_N: int = 256,
    sm_scale: float | None = None,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    """Flash attention that reads a paged KV cache in place (varlen ABI).

    Same math as :func:`tilelang.tpu.kernels.attention.flash_attention_gqa`, but
    the driver no longer materializes a padded ``[B, Hkv, kv_pad, D]`` K/V batch:
    the kernel gathers its own K/V rows out of the cache with ``sdma::gather_h``,
    one ``block_N x D`` tile at a time:

        Q     [Tk, Hq, D]         packed queries of the whole batch, unchanged
        Kc/Vc [Slots, Hkv, D]     the paged cache exactly as stored (NHD)
        Idx   [Bk, Hkv, KV]       uint32 rows of ``[Slots*Hkv, D]`` to gather
        Mask  [Bk, QP, KV]        fp32 additive mask, block-diagonal per request
        QOff  [Bk]                first query row of each request in ``Q``
        Out   [Bk, QP, Hq, D]     per-request padded result

    ``Idx[b, h, j]`` is ``slot * Hkv + h`` for the ``j``-th key of request ``b``
    (and a valid in-range row for padded columns, which the mask discards), so a
    gather of ``block_N`` consecutive entries is one ``block_N x D`` tile read.

    Q and Out are addressed with a *runtime* row offset per request
    (``QOff[b]``, read on device), which is what removes the driver-side pack
    and scatter: ``Q`` is already ``[T, Hq, D]`` in the model's layout.  Tiles
    past a request's own length read or produce garbage rows but never write
    outside its ``[QP]`` slab, so they are harmless.

    ``QP``/``KV`` are bucket-wide (every request in a launch shares them); ``KV``
    must be a multiple of ``block_N`` and ``QP`` of ``block_M``.
    """
    D = int(d)
    scale = float(sm_scale) if sm_scale is not None else 1.0 / (D ** 0.5)

    @T.prim_func
    def main(
        Q: T.Tensor((Tk, Hq, D), dtype),
        Kc: T.Tensor((Slots, Hkv, D), dtype),
        Vc: T.Tensor((Slots, Hkv, D), dtype),
        Idx: T.Tensor((Bk, Hkv, KV), "uint32"),
        Mask: T.Tensor((Bk, QP, KV), accum_dtype),
        QOff: T.Tensor((Bk,), "int32"),
        Out: T.Tensor((Bk, QP, Hq, D), dtype),
    ):
        Tk, Hq, _ = Q.shape
        Slots, Hkv, _ = Kc.shape
        Bk, Hkv, KV = Idx.shape
        _, QP, _ = Mask.shape
        groups = Hq // Hkv

        Tq = T.ceildiv(QP, block_M)
        total_tiles = Bk * Tq
        tiles_per_core = T.ceildiv(total_tiles, _NUM_CORES)

        with T.Kernel(total_tiles, Hq, _NUM_CORES) as (bx, by, bc):
            # 3-D tiles: a [block_M, D] slab of Q / Out spans axes 0 and 2 of a
            # 3-D buffer, so the tile is declared (block_M, 1, D) to make the
            # region extents line up with those axes.  On chip it is the same
            # {1, block_M, 1, D} matrix, so the Q @ K^T gemm takes it through
            # ``gemm_tiled`` (the generic ``T.gemm`` rejects a non-unit leading
            # axis).
            Q_local = T.alloc_local((block_M, 1, D), dtype)
            O_local = T.alloc_local((block_M, 1, D), dtype)
            K_local = T.alloc_local((block_N, D), dtype)
            V_local = T.alloc_local((block_N, D), dtype)
            Idx_local = T.alloc_local((1, block_N), "uint32")
            Mask_local = T.alloc_local((block_M, block_N), accum_dtype)
            acc_s = T.alloc_local((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_local((block_M, block_N), dtype)
            # Also a (rows, 1, D) tile: it is the gemm accumulator *and* the
            # source of the store into ``Out[b, rows, h, :]``, so it must carry
            # the same axes as the global (see the Q_local note above).
            acc_o = T.alloc_local((block_M, 1, D), accum_dtype)
            m_prev = T.alloc_local((block_M,), accum_dtype)
            m_cur = T.alloc_local((block_M,), accum_dtype)
            scale_f = T.alloc_local((block_M,), accum_dtype)
            row_sum = T.alloc_local((block_M,), accum_dtype)
            logsum = T.alloc_local((block_M,), accum_dtype)

            bx_global = bc * tiles_per_core + bx
            with T.If(bx_global < total_tiles), T.Then():
                b = bx_global // Tq
                tq = bx_global % Tq
                hkv = by // groups
                # First query row of this request, read on device: this is what
                # lets one packed [T, Hq, D] buffer serve every request.
                q_off = T.call_extern("int32", "tl_tpu_gmem_read_i32", QOff.data, b)
                # True per-request row of this tile.  The trailing tiles of a
                # request whose q_len is not a multiple of block_M read on into
                # the next request's rows (or the tail padding), producing output
                # for rows the driver discards -- harmless, and the real rows
                # (< q_len) are read from the correct offset so mask / Q / Out
                # stay aligned.  The driver pads Q by block_M rows so the last
                # request's over-read stays in bounds (the overshoot is always
                # < block_M); clamping instead would misalign Q against the mask
                # and output, which corrupts ragged and decode batches.
                q_row = q_off + tq * block_M

                T.copy(Q[q_row, by, 0], Q_local)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(m_cur, -30000.0)

                for k in T.serial(KV // block_N):
                    T.copy(Idx[b, hkv, k * block_N], Idx_local)
                    T.call_extern(
                        "handle", "tl_tpu_gather_rows", Kc.data, Idx_local.data, K_local.data
                    )
                    T.call_extern(
                        "handle", "tl_tpu_gather_rows", Vc.data, Idx_local.data, V_local.data
                    )
                    T.copy(Mask[b, tq * block_M, k * block_N], Mask_local)
                    # S = Q @ K^T  (overwrite acc_s: fresh scores each kv block)
                    gemm_tiled(Q_local, K_local, acc_s, transpose_B=True, clear_accum=True)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = acc_s[i, j] * scale + Mask_local[i, j]
                    T.copy(m_cur, m_prev)
                    T.reduce_max(acc_s, m_cur, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scale_f[i] = T.exp(m_prev[i] - m_cur[i])
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp(acc_s[i, j] - m_cur[i])
                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scale_f[i] + row_sum[i]
                    for i, j in T.Parallel(block_M, D):
                        acc_o[i, 0, j] = acc_o[i, 0, j] * scale_f[i]
                    T.copy(acc_s, acc_s_cast)
                    gemm_tiled(acc_s_cast, V_local, acc_o)

                for i, j in T.Parallel(block_M, D):
                    acc_o[i, 0, j] = acc_o[i, 0, j] / logsum[i]
                T.copy(acc_o, O_local)
                T.copy(O_local, Out[b, tq * block_M, by, 0])

    return main
