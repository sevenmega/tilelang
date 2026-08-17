"""PPL kernel cache for the tilelang JIT pipeline.

For v1 this disables disk caching — PPL kernels are always recompiled. The
in-memory singleton cache from KernelCache still works (same JITKernel instance
is returned for repeated calls with the same func/target/out_idx).
"""

from __future__ import annotations

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class PPLKernelCache(KernelCache):
    kernel_lib_path = "kernel.so"

    def _save_kernel_to_disk(self, key: str, kernel: JITKernel, func=None, verbose: bool = False):
        pass

    def _load_kernel_from_disk(self, key, target=None, target_host=None, out_idx=None,
                               execution_backend=None, pass_configs=None,
                               compile_flags=None, func=None, verbose=False) -> JITKernel | None:
        return None

    def _save_wrapper_kernel_code_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        pass

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        pass
