"""PPL kernel cache for the tilelang JIT pipeline.

Two cache layers sit under a PPL ``.compile()`` / factory call:

1. **In-memory** (this class, via :class:`KernelCache`): repeated calls with the
   same ``(func, target, out_idx, ...)`` key return the *same* ``JITKernel``
   instance from ``_memory_cache`` — no recompile, no re-emit.
2. **On-disk, content-hashed workdir** (``ppl_runner._build_common``): the
   ppl-compiled ``libkernel.so`` + ctypes wrapper ``.so`` live in
   ``/tmp/tilelang_tpu_<kernel>_<md5(emitted source)>/`` and are reused whenever
   the emitted ``.pl`` matches (logged as "build() cache HIT"). This is what
   makes the expensive step — ppl-compile + cmake, seconds — a cross-process
   disk hit; a cold process only re-runs ``emit_pl`` + ``build_generic`` (TIR
   walk + file-existence checks, milliseconds).

The base :class:`KernelCache` disk methods are intentionally **not** reused: they
persist/reload a TVM-runtime kernel (``kernel.adapter.libpath``, a host/device
kernel-source + ``kernel_lib.so`` triple reloaded through ``_build_kernel``).
``PPLKernelAdapter`` is a standalone ctypes wrapper with none of those attributes,
so the base save would ``AttributeError`` and the base load would reconstruct a
broken (TVM-shaped) kernel. Overriding them to no-ops routes PPL through the two
layers above, which together already give correct in-process memoization and
cross-process artifact reuse.

(A future enhancement could copy the two ``.so`` files into the persistent
tilelang cache dir so they survive a ``/tmp`` wipe across reboots; that needs a
PPL-aware ``_load_kernel_from_disk`` that reconstructs a ``TPUKernel`` +
``PPLKernelAdapter`` pointing at the reloaded libs, rather than the base
``_build_kernel``. Not required for correctness — the workdir hit covers a work
session — so it is left out here.)
"""

from __future__ import annotations

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class PPLKernelCache(KernelCache):
    kernel_lib_path = "kernel.so"

    # Disk persistence is a no-op on purpose: the base-class format is
    # TVM-runtime-shaped and does not fit PPL's ctypes adapter. Cross-process
    # reuse of the compiled artifacts is handled by the content-hashed workdir in
    # ppl_runner._build_common (see module docstring); in-process reuse is handled
    # by KernelCache._memory_cache. So load always misses (falling through to a
    # PPL compile that hits the workdir cache) and save writes nothing.

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
