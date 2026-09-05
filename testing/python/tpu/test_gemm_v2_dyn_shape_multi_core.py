"""Multi-core dynamic-shape GEMM+ReLU test for the TPU backend.

Extends test_gemm_v1_dyn_shape with a 3rd T.Kernel axis representing the
TPU core dimension.  The M dimension is partitioned across ``core_num``
cores (SG2260E has 4); each core computes its slice of output rows
independently (no inter-core sync needed).

The 3-axis T.Kernel grid:
  - bx: N-tile index (all cores iterate the full N dimension)
  - by: M-tile index within this core's slice
  - bc: core index (maps to get_block_index() in emitted PPL)
"""
import tilelang
import tilelang.language as T


def matmul_multicore(
    M,
    N,
    K,
    block_M=64,
    block_N=64,
    block_K=32,
    core_num=4,
    in_dtype="float16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), in_dtype),
    ):
        M_tiles_per_core = T.ceildiv(T.ceildiv(M, block_M), core_num)
        with T.Kernel(
            T.ceildiv(N, block_N),  # bx: N tiles
            M_tiles_per_core,       # by: M tiles per core
            core_num,               # bc: core index
        ) as (bx, by, bc):
            A_local = T.alloc_local((block_M, block_K), in_dtype)
            B_local = T.alloc_local((block_K, block_N), in_dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)

            by_global = bc * M_tiles_per_core + by
            with T.If(by_global * block_M < M), T.Then():
                T.clear(C_local)

                for k in T.serial(T.ceildiv(K, block_K)):
                    T.copy(A[by_global * block_M, k * block_K], A_local)
                    T.copy(B[k * block_K, bx * block_N], B_local)
                    T.gemm(A_local, B_local, C_local)

                for i, j in T.Parallel(block_M, block_N):
                    C_local[i, j] = T.max(C_local[i, j], 0)

                T.copy(C_local, C[by_global * block_M, bx * block_N])

    return main


M = T.dynamic("M")
N = T.dynamic("N")
K = T.dynamic("K")

CORE_NUM = 4

program = matmul_multicore(M, N, K, block_M=64, block_N=64, block_K=32,
                           core_num=CORE_NUM)
kernel = tilelang.compile(program, target="tpu", out_idx=[2])
print(f"Multi-core (core_num={CORE_NUM}) dynamic-shape GEMM+ReLU compilation succeeded.")
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
            print(f"\n--- Testing M={M_val}, N={N_val}, K={K_val} (core_num={CORE_NUM}) ---")
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

        print(f"\nAll multi-core dynamic shape tests passed (core_num={CORE_NUM}).")
