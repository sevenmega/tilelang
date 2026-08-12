"""PPL kernel adapter for the tilelang JIT pipeline.

Wraps a compiled TPUKernel so that JITKernel can treat it identically to any
other backend adapter (call it, query source, get profiler, etc.).

This does NOT inherit BaseKernelAdapter because that class assumes a TVM
runtime module (rt_mod / host_mod / device_mod).  PPL is a standalone toolchain.
"""

from __future__ import annotations

from typing import Any

from tilelang.engine.param import KernelParam
from tilelang.tpu.compiler import TPUKernel


class PPLKernelAdapter:
    """Adapter wrapping a PPL-compiled TPUKernel for tilelang JIT."""

    def __init__(
        self,
        tpu_kernel: TPUKernel,
        params: list[KernelParam],
        result_idx: list[int],
    ):
        self._tpu_kernel = tpu_kernel
        self.params = params
        self.result_idx = result_idx
        self.func = self._call

    def _call(self, *args: Any) -> Any:
        if len(args) >= 2:
            return self._tpu_kernel(args[0], args[1])
        raise ValueError("PPL GEMM kernel expects at least 2 tensor arguments (A, B)")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self._tpu_kernel.get_kernel_source(kernel_only=kernel_only)

    def get_host_source(self) -> str:
        return "// PPL backend: no separate host source"

    def get_profiler(self, **kwargs: Any):
        return self._tpu_kernel.get_profiler(**kwargs)

    def get_exportable_executable(self):
        raise NotImplementedError("PPL backend does not support TVM module export")
