# TileLang TPU Backend

This directory implements the TPU (SG2260E) backend for tilelang, using the PPL toolchain to compile and run GEMM kernels on Sophgo TPU hardware.

## Prerequisites

- Python 3.10+
- PyTorch (CPU is sufficient for codegen; TPU runtime needs host tensors)
- PPL release package v1.7.198+ (e.g. `ppl_v1.7.198-gcf5b037f-20260722/`)
- Sophgo TPU device (devid 3 by default)

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
    relu=True, in_dtype="fp16", device=3,
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

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device`  | 3       | TPU device ID for `tpuRtKernelLaunch` |
| `chip`    | sg2260e | Target chip (maps to `tpub_7_1_e`) |
| `in_dtype`| fp16    | Input dtype: `fp16` or `bf16` |
| `block_m` | 64      | M-dimension tile size |
| `block_k` | 32      | K-dimension tile size |
| `block_n` | 64      | N-dimension tile size |
