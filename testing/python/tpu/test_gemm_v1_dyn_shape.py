"""Dynamic-shape GEMM+ReLU test for the TPU backend.

Demonstrates a single kernel definition that supports dynamic M, N, K.
The PPL __KERNEL__ function takes M, K, N as runtime int arguments, so
one compiled kernel handles any shape whose dimensions are divisible by
the tile sizes.

Uses ``@tilelang.jit(target="tpu")`` with ``T.const()`` for compile-time
shape specialization.  The resulting TIR has only 2 spatial levels
(``bx``, ``by``) — no GPU-specific ``tx``/``ty``/``tz`` thread bindings.
"""
import tilelang
import tilelang.language as T


@tilelang.jit(target="tpu")
def matmul_dyn_shape(
    A, B,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
        A_local = T.alloc_local((block_M, block_K), dtype)
        B_local = T.alloc_local((block_K, block_N), dtype)
        C_local = T.alloc_local((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for k in T.serial(T.ceildiv(K, block_K)):
            T.copy(A[by * block_M, k * block_K], A_local)
            T.copy(B[k * block_K, bx * block_N], B_local)
            T.gemm(A_local, B_local, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M, N, K = 1024, 1024, 1024

print("TIR:")
print(matmul_dyn_shape.get_tir(M=M, N=N, K=K).script())

kernel = matmul_dyn_shape.compile(M=M, N=N, K=K)
print("\nDynamic-shape GEMM+ReLU compilation for TPU target succeeded.")

if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        import os
        import torch
        do_profile = "--profile" in sys.argv

        test_shapes = [
            (1024, 1024, 1024),
            (1024, 512, 1024),
        ]

        for M_val, N_val, K_val in test_shapes:
            print(f"\n--- Testing M={M_val}, N={N_val}, K={K_val} ---")
            if do_profile:
                workdir = kernel.adapter._tpu_kernel.paths["workdir"]
                profile_dir = os.path.join(workdir, f"profiling_{M_val}_{N_val}_{K_val}")
                kernel.adapter.enable_profile(profiling_dir=profile_dir)

            a = torch.randn(M_val, K_val, dtype=torch.float16)
            b = torch.randn(K_val, N_val, dtype=torch.float16)
            c = kernel(a, b)
            ref = torch.relu(a.float() @ b.float()).half()
            if torch.allclose(c, ref, rtol=1e-2, atol=1e-2):
                print(f"PASS: TPU result matches torch reference (M={M_val}, N={N_val}, K={K_val})")
            else:
                max_diff = (c - ref).abs().max().item()
                print(f"FAIL: max diff = {max_diff} (M={M_val}, N={N_val}, K={K_val})")
                sys.exit(1)

            if do_profile:
                print(f"\n--- Profiling M={M_val}, N={N_val}, K={K_val} ---")
                kernel.adapter.collect_profile(verbose="--verbose" in sys.argv)

        print("\nAll dynamic shape tests passed.")
