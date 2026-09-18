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
``groups = Hq // Hkv``.  The attention scale ``sm_scale`` (default ``1/sqrt(D)``) is a
plain function argument baked into the TIR at trace time; override it per compile via
``.compile(sm_scale=...)`` / ``.get_tir(sm_scale=...)``.
"""

import tilelang
import tilelang.language as T


@tilelang.jit(target="tpu")
def flash_attention_gqa(
    Q, K, V, Mask,
    block_M: int = 8,
    block_N: int = 128,
    sm_scale: float = 0.08838834764831845,  # 1/sqrt(128)
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    B, Sq, Skv, Hq, Hkv, D = T.const("B, Sq, Skv, Hq, Hkv, D")
    groups = Hq // Hkv

    Q: T.Tensor((B, Hq, Sq, D), dtype)
    K: T.Tensor((B, Hkv, Skv, D), dtype)
    V: T.Tensor((B, Hkv, Skv, D), dtype)
    Mask: T.Tensor((B, Sq, Skv), accum_dtype)
    Out = T.empty((B, Hq, Sq, D), dtype)

    with T.Kernel(T.ceildiv(Sq, block_M), Hq, B) as (bx, by, bz):
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

        T.copy(Q[bz, by, bx * block_M, 0], Q_local)
        T.fill(acc_o, 0)
        T.fill(logsum, 0)
        T.fill(m_cur, -30000.0)

        for k in T.serial(T.ceildiv(Skv, block_N)):
            T.copy(K[bz, by // groups, k * block_N, 0], K_local)
            T.copy(V[bz, by // groups, k * block_N, 0], V_local)
            T.copy(Mask[bz, bx * block_M, k * block_N], Mask_local)
            # S = Q @ K^T  (overwrite acc_s: fresh scores each kv block)
            T.gemm(Q_local, K_local, acc_s, transpose_B=True, clear_accum=True)
            # scale + additive mask
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = acc_s[i, j] * sm_scale + Mask_local[i, j]
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
        T.copy(acc_o, Out[bz, by, bx * block_M, 0])

    return Out
