import itertools
import tilelang
import tilelang.language as T
from tilelang.autotuner import set_autotune_inputs


def get_configs():
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [32]
    num_stages = [3]
    return [
        {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": ns}
        for bm, bn, bk, ns in itertools.product(block_M, block_N, block_K, num_stages)
    ]


def ref_program(A, B):
    import torch
    return torch.relu(A @ B).to(dtype=torch.float32)


@tilelang.autotune(configs=get_configs(), ref_prog=ref_program)
@tilelang.jit(target="tpu")
def matmul_dyn_shape_autotuned(
    A, B,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), accum_dtype)

    with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N)) as (bx, by):
        A_local = T.alloc_local((block_M, block_K), dtype)
        B_local = T.alloc_local((block_K, block_N), dtype)
        C_local = T.alloc_local((block_M, block_N), accum_dtype)

        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[bx * block_M, k * block_K], A_local)
            T.copy(B[k * block_K, by * block_N], B_local)
            T.gemm(A_local, B_local, C_local)

        for i, j in T.Parallel(block_M, block_N):
            C_local[i, j] = T.max(C_local[i, j], 0)

        T.copy(C_local, C[bx * block_M, by * block_N])

    return C


if __name__ == "__main__":
    import sys
    import torch

    M_val, N_val, K_val = 1024, 1024, 1024

    a = torch.randn(M_val, K_val, dtype=torch.float16)
    b = torch.randn(K_val, N_val, dtype=torch.float16)

    print(f"Autotuning GEMM+ReLU for M={M_val}, N={N_val}, K={K_val} ...")
    print(f"Search space: {len(get_configs())} configs")

    with set_autotune_inputs([a, b]):
        kernel = matmul_dyn_shape_autotuned.compile(a, b)

    print(f"\nBest config: {kernel.config}")
    print(f"Best latency: {kernel.latency} ms")
    print(f"Ref latency:  {kernel.ref_latency} ms")

    print("\nPPL Source (best config):")
    print(kernel.get_kernel_source())

    if "--run" in sys.argv:
        do_profile = "--profile" in sys.argv
        if do_profile:
           kernel.adapter.enable_profile()

        c = kernel(a, b)
        c_ref = torch.relu(a @ b).to(dtype=torch.float32)

        if torch.allclose(c, c_ref, rtol=1e-2, atol=1e-2):
            print("\nPASS: TPU result matches torch reference")
        else:
            max_diff = (c - c_ref).abs().max().item()
            print(f"\nFAIL: max diff = {max_diff}")
            sys.exit(1)

        if do_profile:
            kernel.adapter.collect_profile(verbose="--verbose" in sys.argv)

