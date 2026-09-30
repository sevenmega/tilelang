"""Flash-attention (online softmax, GQA) as a real tilelang DSL kernel for TPU.

This is the single source of truth for the flash-attention kernel: it is a plain
``@tilelang.jit(target="tpu")`` prim_func that lowers to TIR and is translated to
PPL op-by-op by ``tilelang.tpu.ppl_codegen`` (no hand-written PPL template).

Layout is **head-major** so the ``[block_M, D]`` / ``[block_N, D]`` local tiles map
onto the *trailing two* axes of each global buffer (tilelang ``T.copy`` right-aligns
a lower-rank local tile onto the trailing buffer axes):

    Q, Out : [B, Hq, Sq, D]      (fp16)
    K, V   : [B, Hkv, Skv, D]    (fp16)
    Mask   : [B, Sq, Skv]        (fp32 additive, fp16-safe ``-30000`` masked / 0 kept)

GQA is expressed directly in the K/V load index as ``by // groups`` where
``groups = Hq // Hkv``.

Dynamic vs baked dims
---------------------
``B, Hq, Sq, Hkv, Skv`` are **dynamic**: declared with ``T.dynamic`` at module scope
and rebound from the input shapes inside ``main``, so a single build serves every
batch / head-count / sequence-length combination.  ``D`` (head dim) and the tile
sizes ``block_M`` / ``block_N`` stay **baked** — they set the shapes of the on-chip
tiles, which the PPL translator requires to be compile-time constants
(``emit_local_allocs`` freezes tile shapes to concrete ints).  The scale
``sm_scale`` (default ``1/sqrt(D)``) is baked into the TIR at trace time.

Because tile extents are baked and *not* clamped to partial tiles, ``Sq`` must be a
multiple of ``block_M`` and ``Skv`` a multiple of ``block_N``; callers pad up to
those multiples (padded key columns get a ``-30000`` additive mask so they
contribute nothing; padded query rows produce throwaway output the caller slices
off).  This is why the sglang path still pads host-side even though the kernel is
shape-generic — padding to a block multiple is a single compiled kernel, not one
build per bucket.

Grid / core mapping
-------------------
``B`` must **not** be its own grid axis: the emitter treats the *last* grid axis
as the core index (``ppl_codegen`` lowers a 3-axis ``T.Kernel`` to
``get_block_index()`` + a guard), and SG2260E has only ``num_cores`` of them.  A
trailing ``B`` axis therefore silently caps the computed batch at ``num_cores``
and leaves the rest of ``Out`` unwritten.  Instead the ``(batch, query-tile)``
pair is flattened into the leading axis, ``total_tiles = B * ceildiv(Sq, block_M)``,
and split across cores; the last axis is ``num_cores`` and the body recovers
``b``/``tq`` from the work index.  The extent guard keeps the kernel correct even
when the emitter does not recognize the per-core partition (it then falls back to
every core sweeping the full range, which duplicates writes of identical values
rather than dropping any).
"""

import tilelang
import tilelang.language as T

# Dynamic shape variables. This module-scope declaration is mandatory: without it
# the nested prim_func raises ``NameError: name 'Sq' is not defined``. D (head dim)
# is deliberately NOT dynamic -- it sizes the on-chip tiles, which must be baked.
B = T.dynamic("B", "int32")
Sq = T.dynamic("Sq", "int32")
Skv = T.dynamic("Skv", "int32")
Hq = T.dynamic("Hq", "int32")
Hkv = T.dynamic("Hkv", "int32")

# SG2260E has four cores; the last grid axis is the core index (see the module
# docstring's "Grid / core mapping"). Baked, like the tile sizes.
_NUM_CORES = 4


@tilelang.jit(target="tpu")
def flash_attention_gqa(
    d: int = 128,
    block_M: int = 8,
    block_N: int = 128,
    sm_scale: float | None = None,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    D = int(d)
    scale = float(sm_scale) if sm_scale is not None else 1.0 / (D ** 0.5)

    @T.prim_func
    def main(
        Q: T.Tensor((B, Hq, Sq, D), dtype),
        K: T.Tensor((B, Hkv, Skv, D), dtype),
        V: T.Tensor((B, Hkv, Skv, D), dtype),
        Mask: T.Tensor((B, Sq, Skv), accum_dtype),
        Out: T.Tensor((B, Hq, Sq, D), dtype),
    ):
        # Bind the dynamic extents from the input shapes (D stays the baked int).
        B, Hq, Sq, _ = Q.shape
        _, Hkv, Skv, _ = K.shape
        groups = Hq // Hkv

        # Flatten (batch, query-tile) into one work axis and split it across cores.
        # The axis carries the FULL work count and ``tiles_per_core`` is exactly
        # ``ceildiv(total_tiles, num_cores)`` -- that is the shape the emitter
        # recognizes as a core partition, and only then does it narrow each core to
        # ``[bc*tiles_per_core, +tiles_per_core)``.  A core whose range starts past
        # ``total_tiles`` gets an *empty* loop, so it never enters the body (and so
        # never pays for the local-tile descriptor setup below).  Putting the
        # already-divided extent on the axis instead makes the recognition fail and
        # every core sweep a non-empty range -- correct, but idle cores then cost
        # ~65us of dead descriptor setup, which dominates small shapes.
        Tq = T.ceildiv(Sq, block_M)
        total_tiles = B * Tq
        tiles_per_core = T.ceildiv(total_tiles, _NUM_CORES)

        with T.Kernel(total_tiles, Hq, _NUM_CORES) as (bx, by, bc):
            Q_local = T.alloc_local((block_M, D), dtype)
            K_local = T.alloc_local((block_N, D), dtype)
            V_local = T.alloc_local((block_N, D), dtype)
            Mask_local = T.alloc_local((block_M, block_N), accum_dtype)
            acc_s = T.alloc_local((block_M, block_N), accum_dtype)
            acc_s_cast = T.alloc_local((block_M, block_N), dtype)
            acc_o = T.alloc_local((block_M, D), accum_dtype)
            m_prev = T.alloc_local((block_M,), accum_dtype)
            m_cur = T.alloc_local((block_M,), accum_dtype)
            scale_f = T.alloc_local((block_M,), accum_dtype)
            row_sum = T.alloc_local((block_M,), accum_dtype)
            logsum = T.alloc_local((block_M,), accum_dtype)

            bx_global = bc * tiles_per_core + bx
            # Guard for the un-narrowed fallback; with the partition recognized the
            # loop bound already clamps, so this never rejects.
            with T.If(bx_global < total_tiles), T.Then():
                b = bx_global // Tq
                tq = bx_global % Tq

                T.copy(Q[b, by, tq * block_M, 0], Q_local)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(m_cur, -30000.0)

                for k in T.serial(T.ceildiv(Skv, block_N)):
                    T.copy(K[b, by // groups, k * block_N, 0], K_local)
                    T.copy(V[b, by // groups, k * block_N, 0], V_local)
                    T.copy(Mask[b, tq * block_M, k * block_N], Mask_local)
                    # S = Q @ K^T  (overwrite acc_s: fresh scores each kv block)
                    T.gemm(Q_local, K_local, acc_s, transpose_B=True, clear_accum=True)
                    # scale + additive mask
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = acc_s[i, j] * scale + Mask_local[i, j]
                    # online softmax running max
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
                        acc_o[i, j] = acc_o[i, j] * scale_f[i]
                    # O += P @ V  (accumulate across kv blocks)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_local, acc_o)

                for i, j in T.Parallel(block_M, D):
                    acc_o[i, j] = acc_o[i, j] / logsum[i]
                T.copy(acc_o, Out[b, by, tq * block_M, 0])

    return main
