---
name: tilelang-tpu-codegen
description: Use when working on the tilelang → PPL → SG2260E TPU code generation pipeline. Covers the PPL kernel template, build system, ctypes wrapper, host launch code, and how tilelang TIR maps to emitted PPL .pl source.
---

# TileLang TPU Codegen Pipeline

## Architecture Overview

The TPU codegen path is **template-based**, not a TIR-walking code emitter. The
pipeline extracts parameters from the tilelang TIR (shapes, tile sizes, dtype,
relu, core_num) and fills a PPL `.pl` template string. The key flow:

```
tilelang DSL (@T.prim_func)
  → TIR PrimFunc (T.Kernel, T.alloc_local, T.gemm, T.copy, etc.)
  → compiler.py: extract M/K/N, block_m/k/n, in_dtype, relu, core_num from TIR
  → PPLGemmSpec dataclass
  → emit_pl(spec) → .pl source string from _PL_TEMPLATE or _PL_TEMPLATE_MULTICORE
  → ppl-compile binary → libkernel.so (device) + host/*.cpp + include/*.h
  → append ctypes wrapper .cpp → cmake → lib<kernel>_py.so
  → PPLKernel loads .so via ctypes, manages device lifecycle
```

## Key Files

| File | Role |
|------|------|
| `tilelang/tpu/ppl_runner.py` | PPL templates, build engine, `PPLKernel` (ctypes runtime), `PPLGemmSpec` |
| `tilelang/tpu/compiler.py` | `compile()` / `compile_gemm()` entry points, TIR parameter extraction |
| `tilelang/tpu/adapter.py` | `PPLKernelAdapter` — bridges `TPUKernel` to tilelang's `JITKernel` |
| `tilelang/jit/kernel.py:274-291` | JIT path: detects `target="tpu"`, calls `tpu_compile()`, wraps in adapter |

## PPL Template System

Four templates in `ppl_runner.py`:

- **`_PL_TEMPLATE`** — Single-core GEMM. Sequential M×N tile loops, `enable_pipeline()`.
- **`_PL_TEMPLATE_MULTICORE`** — Multi-core GEMM. Adds `set_block_num_max()` /
  `get_block_num()` / `get_block_index()` and partitions M across cores.
- **`_PL_TEMPLATE_MULTIBUF`** — Single-core double-buffer. Explicit ping-pong
  buffers (`sub_left_0`/`sub_left_1`) with `parallel_start()`/`parallel_end()`.
- **`_PL_TEMPLATE_MULTICORE_MULTIBUF`** — Multi-core + double-buffer.

All share the same format placeholders:
`{kernel_name}`, `{in_type}` (fp16/bf16), `{block_m}`, `{block_k}`, `{block_n}`,
`{do_relu_int}` (0 or 1), `{M}`, `{K}`, `{N}`.

`emit_pl(spec)` dispatches on `spec.core_num` and `spec.num_stages`:
`num_stages >= 2` → multi-buf variants; `core_num > 1` → multi-core variants.

The `__TEST__` section is used by `ppl-compile` to generate a `test_case` binary.
It uses the default M/K/N from the spec. The `__KERNEL__` takes M/K/N as runtime
int args so one compiled kernel handles any shape divisible by tile sizes.

## PPLGemmSpec Dataclass

```python
@dataclass
class PPLGemmSpec:
    M: int; K: int; N: int
    block_m: int = 64; block_k: int = 64; block_n: int = 64
    relu: bool = True
    in_dtype: str = "fp16"   # "fp16" or "bf16"
    core_num: int = 1        # 1 = single-core, >1 = multi-core
    num_stages: int = 1      # 1 = single-buffer, 2 = double-buffer (explicit ping-pong)
    kernel_name: str = "tl_gemm_relu"
```

Validation: M/K/N must be divisible by their tile sizes (no boundary handling).

## Ctypes Wrapper (.cpp)

`_WRAPPER_TEMPLATE` in `ppl_runner.py` generates a `.cpp` that:
- `py_init_device(devid)` → `tpuRtInit()`, `tpuRtSetDevice()`, `tpuRtStreamCreate()`,
  `tpuRtKernelLoadModuleFile()`, `tpudnnHandleFromStream()` — returns handle
- `py_release_device(h)` → `tpudnnDestroy()`, unload, sync, destroy
- `py_sync_device(h)` → `tpudnnSync()`
- `py_dev_malloc/free/memcpy_h2d/memcpy_d2h` — device memory management
- `py_{kernel_name}(h, res, left, right, M, K, N)` → calls the PPL-generated
  host function which calls `tpudnnLaunchKernel(..., group_num, block_num)`
- `py_enable_profile(h, max_record_num, mode)` → `tpudnnEnableProfile()`
- `py_disable_profile(h)` → `tpudnnDisableProfile()`

The wrapper is built as a shared library appended to the PPL-generated CMakeLists.txt
via `_wrapper_cmake_fragment()`.

## TIR Parameter Extraction (compiler.py)

The codegen **does not walk TIR to emit code**. It extracts parameters only:

- `_shapes_from_func(func)` → (M, K, N) from buffer shapes A:[M,K], B:[K,N]
- `_tiles_from_func(func, in_dtype)` → (block_m, block_k, block_n) from
  `SBlock.alloc_buffers` — the fp32 2D alloc is C_local (block_m, block_n),
  the in_dtype allocs give block_k
- `_detect_relu(func)` → walks TIR for `tir.Max` nodes
- `_in_dtype_from_func(func)` → reads dtype of first buffer
- `_core_num_from_func(func)` → looks for a `bz` For-node with `kind=4`
  (ThreadBinding), whose extent is the core count. Falls back to 1.

All explicit `compile_gemm()` kwargs override IR-derived values.

## Generated Host Code

The PPL compiler (`ppl-compile`) generates host code in `host/<kernel>.cpp`:

```cpp
int tl_gemm_relu(tpudnnHandle_t t_handle, ...) {
    tpu_kernel_api_tl_gemm_relu_t api;
    // ... fill api struct ...
    int group_num = 1;
    int block_num = 1;  // or 4 for multi-core
    ret = tpudnnLaunchKernel(t_handle, "tl_gemm_relu_entry",
                              &api, sizeof(api), group_num, block_num);
}
```

The PPL compiler's `GroupBlockNumAssignPass` MLIR pass auto-detects
`set_block_num_max()` in the `.pl` source and sets `block_num` to
`MAX_TPU_CORE_NUM` (4 for tpub_7_1_e). No manual host code modification needed.

## Build Cache

Build outputs are cached under `/tmp/tilelang_tpu_<kernel>_<M>_<K>_<N>[_mc<N>][_nb<S>]/`.
The `_mc<N>` suffix distinguishes multi-core builds; `_nb<S>` distinguishes multi-buffer. Cache hit: `kernel_so` and
`wrapper_so` both exist → skip compilation.

## PPL Compile Invocation

`_run_ppl_compile()` in `ppl_runner.py` calls the `ppl-compile` binary:

```bash
ppl-compile <pl_file> --chip tpub_7_1_e_rv --mode pcie \
    --dynamic_shape M K N --gen_test -o <workdir>
```

Key flags:
- `--dynamic_shape M K N` — makes M/K/N runtime args in the kernel
- `--gen_test` — generates test_case binary + CMakeLists.txt
- `--autotune` — alternative to `--gen_test`, wraps kernel call in profiling

## Adding a New Kernel Type

To add support for a new kernel beyond GEMM+ReLU:

1. Add a new `PPL<Op>Spec` dataclass (or extend `PPLGemmSpec`)
2. Write a new `_PL_TEMPLATE_<OP>` with the kernel's PPL code
3. Add an `emit_<op>(spec)` function
4. Add a `compile_<op>()` in `compiler.py`
5. The wrapper template is generic — it just calls the generated host function
6. Update `_shapes_from_func` / `_tiles_from_func` if the buffer layout differs
