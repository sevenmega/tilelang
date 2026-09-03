"""Dynamic-shape GEMM+ReLU test for the TPU backend.

Demonstrates a single kernel definition that supports dynamic M, N, K.
The same code structure works on both TPU (target="tpu") and GPU
(target="cuda") — only the target string changes.

Uses T.dynamic() to declare symbolic shape dimensions.  The PPL __KERNEL__
function takes M, K, N as runtime int arguments, so one compiled kernel
handles any shape whose dimensions are divisible by the tile sizes.
"""
import tilelang
import tilelang.language as T


def matmul_dyn_shape(
    M,
    N,
    K,
    block_M=64,
    block_N=64,
    block_K=32,
    in_dtype="float16",
    accum_dtype="float32",
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
            A_local = T.alloc_local((block_M, block_K), in_dtype)
            B_local = T.alloc_local((block_K, block_N), in_dtype)
            C_local = T.alloc_local((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[by * block_M, k * block_K], A_local)
                T.copy(B[k * block_K, bx * block_N], B_local)
                T.gemm(A_local, B_local, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# All three dimensions are dynamic
M = T.dynamic("M")
N = T.dynamic("N")
K = T.dynamic("K")

program = matmul_dyn_shape(M, N, K, block_M=64, block_N=64, block_K=32)
kernel = tilelang.compile(program, target="tpu", out_idx=[2])
print("Dynamic-shape GEMM+ReLU compilation for TPU target succeeded.")

if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        import torch

        test_shapes = [
            (1024, 1024, 1024),
            (1024, 2048, 1024),
        ]

        for M_val, N_val, K_val in test_shapes:
            print(f"\n--- Testing M={M_val}, N={N_val}, K={K_val} ---")
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

        print("\nAll dynamic shape tests passed.")

    if "--profile" in sys.argv:
        from tilelang.tpu.ppl_runner import run_profiling, PPLGemmSpec
        spec = PPLGemmSpec(M=1024, K=1024, N=1024, block_m=64, block_k=32, block_n=64)
        workdir = "/tmp/tilelang_tpu_tl_gemm_relu_dyn_dyn_dyn_profile"
        run_profiling(spec, workdir, verbose="--verbose" in sys.argv)
