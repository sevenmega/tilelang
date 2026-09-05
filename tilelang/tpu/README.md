# TileLang TPU Backend

This directory implements the TPU (SG2260E) backend for tilelang, using the PPL toolchain to compile and run GEMM kernels on Sophgo TPU hardware.

## Prerequisites

- Python 3.10+
- PyTorch (CPU is sufficient for codegen; TPU runtime needs host tensors)
- PPL release package v1.7.198+ (e.g. `ppl_v1.7.198-gcf5b037f-20260722/`)
- Sophgo TPU device (device 0 by default; override with `$TPU_VISIBLE_DEVICES`)

## Environment Setup

Source the PPL environment script before using the TPU backend:

```bash
source /path/to/ppl_v1.7.198-gcf5b037f-20260722/envsetup.sh
```

This sets `PPL_PROJECT_ROOT` and related variables. The TPU backend reads
`PPL_PROJECT_ROOT` to locate:

- `bin/ppl-compile` -- the MLIR compiler binary
- `deps/chip/chip_map.json` -- chip name to arch code mapping
- `deps/chip/tpub_7_1_e/` -- chip-specific libs and configs
- `deps/runtime/tpuv7-runtime/` -- TPU runtime libraries
- `deps/common/` -- host/device utility sources
- `deps/scripts/pcie.cmake`, `GenChipDef.cmake` -- CMake build templates
- `third_party/toolchains_dir/Xuantie-900-gcc-*` -- RISC-V cross-compiler

## Compilation Paths

The build system has two compilation paths, selected automatically:

### Primary path (inline orchestration)

When `$PPL_PROJECT_ROOT/bin/ppl-compile` is found, tilelang drives the
compilation directly:

1. Runs the `ppl-compile` binary to lower `.pl` -> device/host C code
2. Copies `pcie.cmake` as `CMakeLists.txt` and runs cmake + make
3. Appends a ctypes wrapper target and rebuilds

This path does **not** depend on `ppl_compile.py` or its Python dependencies.

### Fallback path (ppl_compile.py subprocess)

If the `ppl-compile` binary is not found (e.g. a source-only PPL checkout),
tilelang falls back to shelling out to `ppl_compile.py --gen_test` as a
subprocess, which is the original compilation method.

## Build

```bash
cd /path/to/tilelang
pip install -e . -v
```

## Run

### Codegen only (no hardware required)

```bash
source /path/to/ppl_.../envsetup.sh
python testing/python/tpu/test_gemm_naive.py
```

This compiles the kernel through the full JIT pipeline and prints the generated PPL `.pl` source.

### Hardware execution (requires TPU + PPL toolchain)

```bash
source /path/to/ppl_.../envsetup.sh
python testing/python/tpu/test_gemm_naive.py --run
```

This compiles the kernel, runs it on the TPU, and validates the result against `torch.relu(A @ B)` with rtol=atol=1e-2.

### Direct compile API (without @jit decorator)

```python
from tilelang.tpu import compile_gemm

kernel = compile_gemm(
    M=1024, K=1024, N=1024,
    block_m=64, block_k=32, block_n=64,
    relu=True, in_dtype="fp16",
)
print(kernel.get_kernel_source())

import torch
a = torch.randn(1024, 1024, dtype=torch.float16)
b = torch.randn(1024, 1024, dtype=torch.float16)
c = kernel(a, b)
```

## Architecture

```
@tilelang.jit(target="tpu")
        |
        v
execution_backend.py  -->  resolves to "ppl" backend
        |
        v
jit/kernel.py         -->  "ppl" branch: skips tilelang.lower()
        |
        v
compiler.py           -->  extracts M/K/N, tiles, relu from PrimFunc
        |
        v
ppl_runner.py         -->  emit_pl() + build():
        |                     primary:  ppl-compile binary + cmake (inline)
        |                     fallback: ppl_compile.py subprocess
        v
adapter.py            -->  PPLKernelAdapter wraps TPUKernel for JIT
```

Key design: the PPL backend bypasses TVM lowering entirely. It pattern-matches the original PrimFunc to extract GEMM shape and tile parameters, then emits native PPL source.

## Supported Operations

Currently: fp16 GEMM with fp32 accumulation, optional ReLU activation, square tiles that divide M/K/N evenly.

## Profiling

Profiling uses the same ctypes execution path as normal kernel runs — no
separate ``test_case`` binary needed.  When profiling is enabled,
``tpudnnEnableProfile()`` is called on the device handle before kernel launch,
so a single run produces both a result tensor (for correctness checking) and
hardware profiling data.

### Quick start

```bash
source /path/to/ppl_.../envsetup.sh

# Correctness test only
python testing/python/tpu/test_gemm_naive.py --run

# Correctness test + profiling in a single execution
python testing/python/tpu/test_gemm_naive.py --run --profile
```

When ``--profile`` is used, the test:

1. Calls ``kernel.enable_profile()`` before the first run
2. Runs the kernel once via ctypes — both correctness and profiling happen
3. Calls ``kernel.collect_profile()`` to process data with ``bigTpuProfile``
4. Prints a summary table and saves detailed data

### Profiling data location

Profiling artifacts are saved in the build workdir:

```
/tmp/tilelang_tpu_tl_gemm_relu_1024_1024_1024/
├── tl_gemm_relu.pl              # PPL kernel source
├── lib/
│   ├── libkernel.so             # device kernel
│   └── tl_gemm_relu_py.so      # ctypes wrapper (includes py_enable_profile)
└── profiling/                   # profiling run output
    ├── cdm_profile_data_dev*    # raw hardware profile data
    └── out_0/
        ├── summary.txt          # parsed summary (Overall time, etc.)
        └── perfetto.pftrace     # Perfetto trace file
```

### Viewing traces in Perfetto

Open the `.pftrace` file in [Perfetto UI](https://ui.perfetto.dev/) for a
detailed timeline view of TIU and DMA operations.  Alternatively, install the
**Perfetto Trace IDE** VS Code extension to view traces directly in the editor.

### Python API

#### Unified path (recommended)

Enable profiling on the kernel object, then run normally.  One execution does
both correctness checking and profiling:

```python
kernel = tilelang.compile(program, target="tpu", out_idx=[2])

# Enable profiling before the run
kernel.adapter.enable_profile()

# Single run — correctness + profiling
c = kernel(a, b)

# Collect profiling data
result = kernel.adapter.collect_profile(verbose=True)
print(result["overall_us"])       # Overall kernel time in microseconds
print(result["pftrace_path"])     # Path to .pftrace file
```

#### Standalone convenience function

For quick one-off profiling without a kernel object:

```python
from tilelang.tpu.ppl_runner import run_profiling, PPLGemmSpec

workdir = "/tmp/tilelang_tpu_tl_gemm_relu_1024_1024_1024"
spec = PPLGemmSpec(M=1024, K=1024, N=1024, block_m=64, block_k=32, block_n=64)
result = run_profiling(workdir, spec=spec)
```

### Environment variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `TPU_VISIBLE_DEVICES` | integer | 0 | TPU device ID (like `CUDA_VISIBLE_DEVICES`). All TPU functions read this as the default device. |
| `PROFILE_BOOK_KEEPING` | 0, 1, 2 | 1 | Profiling detail level passed to `tpudnnEnableProfile()`. Higher values capture more detail at the cost of overhead. |
| `BMLIB_ENABLE_ALL_PROFILE` | 0, 1 | 0 | Set to 1 to enable hardware profiling. Automatically set by ``enable_profile()``. |

### Requirements

- `bigTpuProfile` Python package (`pip install bigTpuProfile`)

## Configuration

### Compile-time parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chip`    | sg2260e | Target chip (maps to `tpub_7_1_e`) |
| `in_dtype`| fp16    | Input dtype: `fp16` or `bf16` |
| `block_m` | 64      | M-dimension tile size |
| `block_k` | 32      | K-dimension tile size |
| `block_n` | 64      | N-dimension tile size |

### Runtime parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device`  | `$TPU_VISIBLE_DEVICES` (default 0) | TPU device ID for `tpuRtKernelLaunch`. Resolved when the kernel is first called, not at compile time. |
