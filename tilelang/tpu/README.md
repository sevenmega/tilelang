# TileLang TPU Backend

This directory implements the TPU (SG2260E) backend for tilelang, using the PPL toolchain to compile and run GEMM kernels on Sophgo TPU hardware.

## Prerequisites

- Python 3.10+
- PyTorch (CPU is sufficient for codegen; TPU runtime needs host tensors)
- PPL toolchain v1.7.198+ (for hardware execution)
- Sophgo TPU device (devid 2 by default)

## Environment Setup

### 1. PPL Toolchain

Source the PPL environment script or set variables manually:

```bash
# Option A: source envsetup.sh
source /path/to/ppl_v1.7.198-gcf5b037f-20260722/envsetup.sh

# Option B: set variables manually
export PPL_PROJECT_ROOT=/path/to/ppl_v1.7.198-gcf5b037f-20260722
export PPL_BUILD_PATH=$PPL_PROJECT_ROOT/build
export PPL_INSTALL_PATH=$PPL_PROJECT_ROOT/install
export PPL_RUNTIME_PATH=$PPL_INSTALL_PATH/lib
export PATH=$PPL_INSTALL_PATH/bin:$PATH
export LD_LIBRARY_PATH=$PPL_RUNTIME_PATH:$LD_LIBRARY_PATH
```

### 2. Verify PPL is available

```bash
which ppl_compile.py
# Should print: /path/to/ppl_.../install/bin/ppl_compile.py
```

## Build

```bash
cd /home/lwang/work/tilelang/tilelang-tpu-new
pip install -e . -v
```

## Run

### Codegen only (no hardware required)

```bash
python testing/python/tpu/test_gemm_naive.py
```

This compiles the kernel through the full JIT pipeline and prints the generated PPL `.pl` source. It will fail at the `ppl_compile.py` subprocess invocation if the PPL toolchain is not installed — this is expected.

### Hardware execution (requires TPU + PPL toolchain)

```bash
python testing/python/tpu/test_gemm_naive.py --run
```

This compiles the kernel, runs it on TPU devid 2, and validates the result against `torch.relu(A @ B)` with rtol=atol=1e-2.

### Direct compile API (without @jit decorator)

```python
from tilelang.tpu import compile_gemm

kernel = compile_gemm(
    M=1024, K=1024, N=1024,
    block_m=64, block_k=32, block_n=64,
    relu=True, in_dtype="fp16", device=2,
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
ppl_runner.py         -->  emit_pl() + build() via ppl_compile.py
        |
        v
adapter.py            -->  PPLKernelAdapter wraps TPUKernel for JIT
```

Key design: the PPL backend bypasses TVM lowering entirely. It pattern-matches the original PrimFunc to extract GEMM shape and tile parameters, then emits native PPL source.

## Supported Operations

Currently: fp16 GEMM with fp32 accumulation, optional ReLU activation, square tiles that divide M/K/N evenly.

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `device`  | 2       | TPU device ID for `tpuRtKernelLaunch` |
| `chip`    | sg2260e | Target chip (maps to `tpub_7_1_e`) |
| `in_dtype`| fp16    | Input dtype: `fp16` or `bf16` |
| `block_m` | 64      | M-dimension tile size |
| `block_k` | 32      | K-dimension tile size |
| `block_n` | 64      | N-dimension tile size |
