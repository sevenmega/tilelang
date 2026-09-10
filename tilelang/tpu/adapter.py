"""PPL kernel adapter for the tilelang JIT pipeline.

Wraps a compiled TPUKernel so that JITKernel can treat it identically to any
other backend adapter (call it, query source, get profiler, etc.).

This does NOT inherit BaseKernelAdapter because that class assumes a TVM
runtime module (rt_mod / host_mod / device_mod).  PPL is a standalone toolchain.
"""

from __future__ import annotations

from typing import Any

import torch
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
        self.result_idx = self._legalize_result_idx(result_idx)
        self.func = self._call

    def _legalize_result_idx(self, result_idx: list[int]) -> list[int]:
        n = len(self.params)
        if n == 0:
            return result_idx
        return [i % n for i in result_idx]

    def _call(self, *args: Any) -> Any:
        n_params = len(self.params)
        n_outputs = len(self.result_idx)
        n_inputs = n_params - n_outputs

        if len(args) == n_inputs:
            # Auto-allocate mode: kernel(a, b) — only inputs provided.
            # Run the kernel and return the result, cast to declared dtype.
            inputs: list[Any] = []
            ins_idx = 0
            for i in range(n_params):
                if i not in self.result_idx:
                    inputs.append(args[ins_idx])
                    ins_idx += 1
            if len(inputs) < 2:
                raise ValueError("PPL GEMM kernel expects at least 2 tensor inputs (A, B)")
            result = self._tpu_kernel(inputs[0], inputs[1])
            if self.result_idx:
                declared_dtype = self.params[self.result_idx[0]].torch_dtype()
                if result.dtype != declared_dtype:
                    result = result.to(declared_dtype)
            return result

        if len(args) == n_params:
            # Write-into mode: kernel(a, b, c) — all params including
            # pre-allocated output tensor(s).  Run the kernel with the
            # input tensors, then copy the result into the caller's
            # output tensor (handles dtype conversion via copy_()).
            inputs = []
            outputs = []
            for i in range(n_params):
                if i in self.result_idx:
                    outputs.append(args[i])
                else:
                    inputs.append(args[i])
            if len(inputs) < 2:
                raise ValueError("PPL GEMM kernel expects at least 2 tensor inputs (A, B)")
            result = self._tpu_kernel(inputs[0], inputs[1])
            for out in outputs:
                out.copy_(result)
            return None

        raise ValueError(
            f"PPL kernel accepts {n_inputs} inputs (auto-allocate output) "
            f"or {n_params} args (write into pre-allocated output), "
            f"but {len(args)} were provided."
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self._tpu_kernel.get_kernel_source(kernel_only=kernel_only)

    def get_host_source(self) -> str:
        return "// PPL backend: no separate host source"

    def enable_profile(self, **kwargs: Any) -> None:
        self._tpu_kernel.enable_profile(**kwargs)

    def collect_profile(self, **kwargs: Any) -> dict:
        return self._tpu_kernel.collect_profile(**kwargs)

    def get_profiler(self, **kwargs: Any):
        return self._tpu_kernel.get_profiler(**kwargs)

    def close(self):
        self._tpu_kernel.close()

    def get_exportable_executable(self):
        raise NotImplementedError("PPL backend does not support TVM module export")
