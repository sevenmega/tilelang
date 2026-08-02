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

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for k in T.serial(T.ceildiv(K, block_K)):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


M, N, K = 1024, 1024, 1024

kernel = matmul_naive.compile(M=M, N=N, K=K)
print("TPU Kernel Source:")
print(kernel.get_kernel_source())
print("\nNaive GEMM+ReLU compilation for TPU target succeeded.")
