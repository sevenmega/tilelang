import tilelang
import tilelang.language as T


@tilelang.jit(target="tpu")
def matmul_naive(
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

print("func_source:")
print(matmul_naive.func_source)
print("signature:")
print(matmul_naive.signature)

print("TIR:")
print(matmul_naive.get_tir(M=M, N=N, K=K).script())

kernel = matmul_naive.compile(M=M, N=N, K=K)
print("\nTPU Kernel Source:")
print(kernel.get_kernel_source())
print("\nNaive GEMM+ReLU compilation for TPU target succeeded.")

# Optional: run on hardware if PPL toolchain is available
if __name__ == "__main__":
    import sys
    if "--run" in sys.argv:
        import torch
        from tilelang.tpu.ppl_runner import ensure_tpu_device

        DEV = f"tpu:{ensure_tpu_device()}"
        do_profile = "--profile" in sys.argv
        if do_profile:
            kernel.adapter.enable_profile()

        a = torch.randn(M, K, dtype=torch.float16, device=DEV)
        b = torch.randn(K, N, dtype=torch.float16, device=DEV)
        # c = matmul_naive(a, b)
        c = kernel(a, b)
        # Reference on CPU: on-device torch matmul is not a valid oracle.
        c_ref = torch.relu(a.cpu().float() @ b.cpu().float()).half()
        c_host = c.cpu()

        # torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        if torch.allclose(c_host, c_ref, rtol=1e-2, atol=1e-2):
            print("PASS: TPU result matches torch reference")
        else:
            max_diff = (c_host - c_ref).abs().max().item()
            print(f"FAIL: max diff = {max_diff}")
            sys.exit(1)

        if do_profile:
            kernel.adapter.collect_profile(verbose="--verbose" in sys.argv)
