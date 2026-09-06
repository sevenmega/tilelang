---
name: tilelang-tpu-dsl-to-ppl
description: Use when writing or modifying tilelang DSL kernels that target the TPU backend. Covers T.Kernel grid semantics, how DSL constructs map to PPL primitives, dynamic shapes, multi-core TIR patterns, and the constraints on what the TPU codegen can handle.
---

# TileLang DSL → TPU/PPL Mapping

## How It Works

The TPU backend uses **template-based codegen**: the tilelang TIR is analyzed for
parameters (not walked to emit code). The DSL kernel is a specification — the
actual PPL `.pl` code comes from fixed templates in `ppl_runner.py`.

This means:
- The DSL kernel must follow a recognized pattern (currently: GEMM + optional ReLU)
- Exotic TIR constructs beyond what the templates support are silently ignored
- The value of the DSL is expressing intent (shapes, tiles, core count) in a
  backend-portable way, not controlling every PPL instruction

## T.Kernel Grid → PPL Core Dispatch

### Single-Core (2-axis grid)

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
    # bx, by → sequential nested loops in emitted PPL
    # for (idx_n = 0; idx_n < N; idx_n += block_n)
    #   for (idx_m = 0; idx_m < M; idx_m += block_m)
```

Emits `_PL_TEMPLATE` with `block_num = 1` in host launch.

### Multi-Core (3-axis grid)

```python
M_tiles_per_core = T.ceildiv(T.ceildiv(M, block_M), core_num)
with T.Kernel(
    T.ceildiv(N, block_N),    # bx: N tiles (all cores)
    M_tiles_per_core,          # by: M tiles per core
    core_num,                  # bc → bz: core index
) as (bx, by, bc):
    by_global = bc * M_tiles_per_core + by
    with T.If(by_global * block_M < M), T.Then():
        # ... tile body ...
```

The 3rd axis becomes a `bz` For-node with `kind=4` (ThreadBinding) in TIR.
`_core_num_from_func()` detects it. Emits `_PL_TEMPLATE_MULTICORE` with
`set_block_num_max()` / `get_block_num()` / `get_block_index()`.

**Key**: The `T.If` guard requires `T.Then()` — use `with T.If(cond), T.Then():`.
Bare `with T.If(cond):` raises `IfThenElse frame should be either in ThenFrame`.

## DSL Construct → PPL Primitive Mapping

| TileLang DSL | PPL Equivalent | Notes |
|-------------|----------------|-------|
| `T.alloc_local((M,K), dtype)` | `tensor<dtype>(shape)` | Local SRAM tile |
| `T.clear(C_local)` | `tiu::zero(sub_res)` | Zero the accumulator |
| `T.copy(A[offset], A_local)` | `dma::load(tile, gtensor.sub_view(...))` | Global → local DMA |
| `T.copy(C_local, C[offset])` | `dma::store(gtensor.sub_view(...), tile)` | Local → global DMA |
| `T.gemm(A, B, C)` | `tiu::fmm2_nn(C, A, B, ..., result_add=true, ...)` | Matrix multiply-accumulate |
| `T.max(val, 0)` (ReLU) | `do_relu=1` on last K iter of `fmm2_nn` | Fused into the matmul |
| `T.serial(N)` | `for (int i = 0; i < N; i++)` | Sequential loop |
| `T.Pipelined(N, num_stages=2)` | Prologue/mainloop/epilogue with `parallel_start`/`parallel_end` | Explicit double-buffer ping-pong |
| `T.Parallel(M, N)` | Element-wise loop (fused into cast/store) | Parallel iteration hint |
| `T.ceildiv(a, b)` | `(a + b - 1) / b` | Ceiling division |
| `T.dynamic("M")` | Runtime `int M` parameter in `__KERNEL__` | Dynamic shape dim |

## Dynamic Shapes

Dynamic dims use `T.dynamic("name")`:

```python
M = T.dynamic("M")
N = T.dynamic("N")
K = T.dynamic("K")
program = matmul(M, N, K, block_M=64, block_N=64, block_K=32)
kernel = tilelang.compile(program, target="tpu", out_idx=[2])
```

The PPL `__KERNEL__` receives M/K/N as `int` parameters. The `__TEST__` uses
concrete default values (1024). At runtime, actual tensor shapes are passed
through the ctypes wrapper.

Constraint: actual M/K/N must be divisible by block_M/K/N (no boundary handling).

## Multi-Buffer Pipeline (T.Pipelined)

### DSL Pattern

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.copy(A[by_global * block_M, k * block_K], A_local)
    T.copy(B[k * block_K, bx * block_N], B_local)
    T.gemm(A_local, B_local, C_local)
```

`num_stages` is stored in `For.annotations["num_stages"]` in TIR.
`_num_stages_from_func()` detects it. `num_stages=1` (or `T.serial`) = no
pipelining; `num_stages=2` = double-buffer with explicit ping-pong.

### Emitted PPL Pattern (num_stages=2)

Instead of PPL's `enable_pipeline()` auto-duplication, the codegen emits
explicit double buffer arrays with manual ping-pong indexing:

```c
// Two sets of input tiles
auto sub_left_0 = tensor<fp16>(left_max_shape);
auto sub_left_1 = tensor<fp16>(left_max_shape);
auto sub_right_0 = tensor<fp16>(right_max_shape);
auto sub_right_1 = tensor<fp16>(right_max_shape);

// Prologue: load first tile pair into buffer 0
dma::load(sub_left_0, ...);
dma::load(sub_right_0, ...);

// Main loop: overlap DMA[i+1] with TIU[i]
int ping = 0;
for (int ki = 0; ki < K_iters - 1; ki++) {
    parallel_start();
    if (ping == 0) {
        dma::load(sub_left_1, ...next...);  // load into buf 1
        tiu::fmm2_nn(sub_res, sub_left_0, sub_right_0, ...);  // compute from buf 0
    } else {
        dma::load(sub_left_0, ...next...);  // load into buf 0
        tiu::fmm2_nn(sub_res, sub_left_1, sub_right_1, ...);  // compute from buf 1
    }
    parallel_end();
    ping = 1 - ping;
}

// Epilogue: compute last iteration (do_relu only here)
```

Key: `parallel_start()`/`parallel_end()` bracket concurrent DMA+TIU regions.
`do_relu` is only applied in the epilogue (last K iteration) to avoid
corrupting the fp32 accumulator on intermediate partial sums.

### Combined Multi-Core + Multi-Buffer

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
```
inside a 3-axis `T.Kernel` produces `_PL_TEMPLATE_MULTICORE_MULTIBUF` —
both M-partitioning across cores AND explicit double-buffering.

## What the Codegen Extracts from TIR

The TPU `compile()` function reads these from the PrimFunc:

1. **Shapes**: `_shapes_from_func()` reads buffer_map → A:[M,K], B:[K,N] → (M, K, N)
2. **Tiles**: `_tiles_from_func()` reads `SBlock.alloc_buffers`:
   - fp32 2D alloc → C_local → (block_m, block_n)
   - in_dtype 2D allocs → A/B shared tiles → block_k
3. **ReLU**: `_detect_relu()` searches for `tir.Max` nodes
4. **Dtype**: `_in_dtype_from_func()` reads first buffer's dtype → fp16 or bf16
5. **Core count**: `_core_num_from_func()` finds `bz` For-node (kind=4) extent
6. **Num stages**: `_num_stages_from_func()` reads `For.annotations["num_stages"]`

All are overridable via explicit kwargs to `compile()` or `compile_gemm()`.

## Test File Patterns

### Compilation-only test (no hardware)

```python
program = matmul(M, N, K, ...)
kernel = tilelang.compile(program, target="tpu", out_idx=[2])
print(kernel.get_kernel_source())  # prints emitted .pl
```

### Hardware correctness test (`--run`)

```python
if "--run" in sys.argv:
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    c = kernel(a, b)
    ref = torch.relu(a.float() @ b.float()).half()
    assert torch.allclose(c, ref, rtol=1e-2, atol=1e-2)
```

### Hardware profiling test (`--profile`)

```python
if "--profile" in sys.argv:
    kernel.adapter.enable_profile(profiling_dir=...)
    c = kernel(a, b)
    kernel.adapter.collect_profile(verbose=True)
```

Profiling requires `BMLIB_ENABLE_ALL_PROFILE=1` set before device init.
`enable_profile()` sets this and forces runtime reinitialization.

## Compilation Output Structure

```
/tmp/tilelang_tpu_<kernel>_<M>_<K>_<N>[_mc<cores>][_nb<stages>]/
  tl_gemm_relu.pl                  # emitted PPL source
  lib/
    libkernel.so                   # device binary (loaded onto TPU)
    tl_gemm_relu_py.so             # ctypes wrapper (loaded by Python)
  host/tl_gemm_relu.cpp            # PPL-generated host launch code
  include/tl_gemm_relu.h           # PPL-generated API struct + function decl
  device/tl_gemm_relu.c            # PPL-generated device code (firmware binary)
  src/tl_gemm_relu.cpp             # PPL-generated test harness
  test_case                        # compiled test binary
  build/                           # cmake build directory
```

## Constraints and Limitations

1. **Only GEMM(+ReLU) supported** — the template-based approach requires a new
   template for each kernel type
2. **No boundary handling** — M/K/N must be divisible by tile sizes
3. **fp16 input only verified** — bf16 path exists but correctness is not validated
4. **Accumulator is always fp32** — hardcoded in templates
5. **4D [1,M,1,N] layout** — PPL uses 4D shapes; the M and N dims are at positions
   1 and 3 (not 0 and 1)
6. **Multi-core partitions M only** — N and K are not split across cores
7. **num_stages=2 only** — explicit double-buffer; triple-buffer (num_stages=3) not yet implemented
8. **No L2 cache optimization** — the reference `mlp_multicore.pl` uses L2
   (`gtensor<fp16>(..., L2)`) for shared data; our template doesn't yet
