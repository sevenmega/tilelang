import tilelang
import tilelang.language as T

# following dynamic variable declaration is mandatory
# otherwise, will see errors like "NameError: name 'M' is not defined"
M = T.dynamic("M", "int32")
N = T.dynamic("N", "int32")
K = T.dynamic("K", "int32")

# T.Buffer() is obsolete, use T.Tensor() instead.
# T.float16 is a shorthand for T.dtype("float16").
@tilelang.jit(target="tpu")
def matmul_dyn_shape_pilelined(
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):

        # these are dynamic shapes
        M, K = A.shape
        _, N = B.shape

        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N)) as (bx, by):
            A_local = T.alloc_local((block_M, block_K), dtype)
            B_local = T.alloc_local((block_K, block_N), dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[bx * block_M, k * block_K], A_local)
                T.copy(B[k * block_K, by * block_N], B_local)
                # T.copy(A[bx * BLOCK_M : (bx + 1) * BLOCK_M, k * BLOCK_K : (k + 1) * BLOCK_K], A_local)
                # T.copy(B[k * BLOCK_K : (k + 1) * BLOCK_K, by * BLOCK_N : (by + 1) * BLOCK_N], B_local)
                T.gemm(A_local, B_local, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[bx * block_M, by * block_N])
            # T.copy(C_local, C[bx * BLOCK_M : (bx + 1) * BLOCK_M, by * BLOCK_N : (by + 1) * BLOCK_N])

    return main

print("func_source:")
print(matmul_dyn_shape_pilelined.func_source)
print("signature:")
print(matmul_dyn_shape_pilelined.signature)

print("TIR:")
print(matmul_dyn_shape_pilelined.get_tir().script())

# kernel = matmul_dyn_shape_pilelined.compile()
kernel = matmul_dyn_shape_pilelined()
print("CUDA Source:")
print(kernel.get_kernel_source())
print("\nDynamic Shape Pipelined GEMM+ReLU compilation for TPU target succeeded.")

if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        import torch
        do_profile = "--profile" in sys.argv
        if do_profile:
            kernel.adapter.enable_profile()

        test_shapes = [
            (1024, 1024, 1024),
            (1024,  512, 1024),
            (2048, 4096, 2048),
        ]

        for M_val, N_val, K_val in test_shapes:
            print(f"\n--- Testing M={M_val}, N={N_val}, K={K_val} ---")
            a = torch.randn(M_val, K_val, dtype=torch.float16)
            b = torch.randn(K_val, N_val, dtype=torch.float16)
            c = torch.empty(M_val, N_val, dtype=torch.float32)
            kernel(a, b, c)
            # float16 by default, change to float32 for comparison
            c_ref = torch.relu(a @ b).to(dtype=torch.float32)

            # torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
            if torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2):
                print("PASS: TPU result matches torch reference")
            else:
                max_diff = (c - c_ref).abs().max().item()
                print(f"FAIL: max diff = {max_diff}")
                print(c)
                print(c_ref)
                sys.exit(1)

            if do_profile:
                kernel.adapter.collect_profile(verbose="--verbose" in sys.argv)

