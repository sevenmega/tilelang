"""Multi-core double-buffered GEMM+ReLU test for the TPU backend.

Extends test_gemm_v2_dyn_shape_multi_core with explicit double-buffering
via ``T.Pipelined(extent, num_stages=2)``.  The K-loop uses a
prologue/mainloop/epilogue pattern with ``parallel_start``/``parallel_end``
to overlap DMA loads into one buffer set while TIU computes on the other.

The 3-axis T.Kernel grid (same as v2):
  - bx: N-tile index (all cores iterate the full N dimension)
  - by: M-tile index within this core's slice
  - bc: core index (maps to get_block_index() in emitted PPL)

Key difference from v2: ``T.Pipelined(extent, num_stages=2)`` replaces
``T.serial(extent)`` for the K-loop.  The codegen detects ``num_stages``
from TIR annotations and emits explicit ping-pong buffer management
instead of relying on PPL's ``enable_pipeline()`` auto-duplication.
"""
import tilelang
import tilelang.language as T


CORE_NUM = 4
NUM_STAGES = 2


@tilelang.jit(target="tpu")
def matmul_multicore_multibuf(
    A, B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    core_num: int = CORE_NUM,
    num_stages: int = NUM_STAGES,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    M_tiles_per_core = T.ceildiv(T.ceildiv(M, block_M), core_num)
    with T.Kernel(
        T.ceildiv(N, block_N),  # bx: N tiles
        M_tiles_per_core,       # by: M tiles per core
        core_num,               # bc: core index
    ) as (bx, by, bc):
        A_local = T.alloc_local((block_M, block_K), dtype)
        B_local = T.alloc_local((block_K, block_N), dtype)
        C_local = T.alloc_local((block_M, block_N), accum_dtype)

        by_global = bc * M_tiles_per_core + by
        with T.If(by_global * block_M < M), T.Then():
            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by_global * block_M, k * block_K], A_local)
                T.copy(B[k * block_K, bx * block_N], B_local)
                T.gemm(A_local, B_local, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by_global * block_M, bx * block_N])

    return C


M, N, K = 1024, 1024, 1024

print("TIR:")
print(matmul_multicore_multibuf.get_tir(M=M, N=N, K=K).script())

kernel = matmul_multicore_multibuf.compile(M=M, N=N, K=K)
print(f"\nMulti-core (core_num={CORE_NUM}) double-buffered (num_stages={NUM_STAGES}) "
      f"GEMM+ReLU compilation succeeded.")
print("\nGenerated PPL kernel source:")
print(kernel.get_kernel_source())

if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        import os
        import torch
        do_profile = "--profile" in sys.argv

        test_shapes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (1024, 512, 1024),
        ]

        for M_val, N_val, K_val in test_shapes:
            print(f"\n--- Testing M={M_val}, N={N_val}, K={K_val} "
                  f"(core_num={CORE_NUM}, num_stages={NUM_STAGES}) ---")
            if do_profile:
                workdir = kernel.adapter._tpu_kernel.paths["workdir"]
                profile_dir = os.path.join(workdir, f"profiling_{M_val}_{N_val}_{K_val}")
                kernel.adapter.enable_profile(profiling_dir=profile_dir)

            a = torch.randn(M_val, K_val, dtype=torch.float16)
            b = torch.randn(K_val, N_val, dtype=torch.float16)
            c = kernel(a, b)
            ref = torch.relu(a.float() @ b.float()).half()
            if torch.allclose(c, ref, rtol=1e-2, atol=1e-2):
                print(f"PASS: TPU result matches torch reference")
            else:
                max_diff = (c - ref).abs().max().item()
                print(f"FAIL: max diff = {max_diff}")
                sys.exit(1)

            if do_profile:
                print(f"\n--- Profiling M={M_val}, N={N_val}, K={K_val} ---")
                kernel.adapter.collect_profile(verbose="--verbose" in sys.argv)

        print(f"\nAll multi-core double-buffered tests passed "
              f"(core_num={CORE_NUM}, num_stages={NUM_STAGES}).")
