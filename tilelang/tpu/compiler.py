"""tilelang TPU compiler facade (generic TIR->PPL path).

Exposes the tilelang-style compile entry point for the SG2260E TPU backed by the
PPL toolchain (see :mod:`tilelang.tpu.ppl_runner`).

``compile(func)`` is **generic**: it walks a *lowered* tilelang ``PrimFunc`` (the
output of ``JITImpl.get_tir`` / a ``@tilelang.jit(target="tpu")`` trace) op-by-op
via :mod:`tilelang.tpu.ppl_codegen` and emits PPL -- no per-family template.  The
same path serves GEMM, flash-attention, and future fused kernels.

The legacy hand-written GEMM(+ReLU) template (``compile_gemm`` / ``PPLGemmSpec``)
is retained for the runtime-int-shape / multi-core / multi-buffer GEMM variants
and for ``run_profiling``; it is *not* on the generic ``.compile()`` path.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import Any

import torch

from tilelang.tpu.ppl_runner import (
    PPLKernel, PPLGenericKernel, PPLGemmSpec, PPLKernelInfo, build, build_generic,
    emit_pl, get_tpu_device, _collect_profile_data,
)
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Profiler stub                                                                #
# --------------------------------------------------------------------------- #


class _TPUProfiler:
    """Minimal profiler: times the kernel launch end-to-end (ms)."""

    def __init__(self, kernel: "TPUKernel", warmup: int = 5, iters: int = 20):
        self._kernel = kernel
        self._warmup = warmup
        self._iters = iters

    def do_bench(self) -> float:
        k = self._kernel
        if k._last_inputs is None:
            raise RuntimeError("call the kernel once before profiling")
        for _ in range(self._warmup):
            k(*k._last_inputs)
        t0 = time.perf_counter()
        for _ in range(self._iters):
            k(*k._last_inputs)
        return (time.perf_counter() - t0) / self._iters * 1e3


# --------------------------------------------------------------------------- #
# Compiled kernel (generic)                                                    #
# --------------------------------------------------------------------------- #


class TPUKernel:
    """A compiled TPU kernel produced by the generic TIR->PPL translator.

    Callable as ``out = kernel(*inputs)`` with CPU torch tensors (one per
    non-output PrimFunc param, in order); inputs are cast to the kernel's
    declared dtypes.  Returns the output tensor (or a tuple for multi-output).
    """

    def __init__(self, info: PPLKernelInfo, paths: dict[str, str], *, device: int | None = None):
        self.info = info
        self.paths = paths
        self._device = device
        self._runtime: PPLGenericKernel | None = None
        self._last_inputs: list[torch.Tensor] | None = None
        self._profile_config: dict[str, Any] | None = None

    @property
    def device(self) -> int:
        if self._device is None:
            self._device = get_tpu_device()
        return self._device

    def _ensure_runtime(self) -> PPLGenericKernel:
        if self._runtime is None:
            rt = PPLGenericKernel(
                self.paths, device=self.device,
                kernel_name=self.info.kernel_name, info=self.info,
            )
            rt.init()
            if self._profile_config is not None:
                cfg = self._profile_config
                rt.enable_profile(cfg["max_record_num"], cfg["book_keeping"])
                rt._profile_dir = cfg["dir"]
            self._runtime = rt
        return self._runtime

    def __call__(self, *inputs: torch.Tensor):
        casted = [t.detach().cpu().contiguous() for t in inputs]
        self._last_inputs = casted
        try:
            return self._ensure_runtime().run(*casted)
        except (RuntimeError, OSError):
            # Device handle may be stale (e.g. invalidated by another config's
            # py_init_device during autotuning).  Reinitialize and retry once.
            self.close()
            return self._ensure_runtime().run(*casted)

    def enable_profile(
        self,
        *,
        max_record_num: int = 0,
        book_keeping: int = 1,
        profiling_dir: str | None = None,
    ) -> None:
        """Enable hardware profiling for subsequent kernel runs.

        Must be called *before* the first ``__call__`` (or after ``close()``).
        """
        os.environ["BMLIB_ENABLE_ALL_PROFILE"] = "1"
        os.environ["PROFILE_BOOK_KEEPING"] = str(book_keeping)
        pd = profiling_dir or os.path.join(self.paths["workdir"], "profiling")
        os.makedirs(pd, exist_ok=True)
        self._profile_config = {
            "max_record_num": max_record_num,
            "book_keeping": book_keeping,
            "dir": pd,
        }
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def collect_profile(self, *, verbose: bool = False) -> dict[str, Any]:
        """Process profiling data from the last profiled run."""
        if self._profile_config is None:
            raise RuntimeError("Profiling not enabled — call enable_profile() first")
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        return _collect_profile_data(self._profile_config["dir"], verbose=verbose)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        return self.info.source

    def get_profiler(self, **_kw: Any) -> "_TPUProfiler":
        return _TPUProfiler(self)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Generic compile entry point                                                  #
# --------------------------------------------------------------------------- #


def compile(
    func: Any,
    *,
    out_idx: int | list[int] = -1,
    target: str = "tpu",
    workdir: str | None = None,
    **build_kw: Any,
) -> TPUKernel:
    """Compile a *lowered* tilelang PrimFunc for the TPU (generic TIR->PPL).

    Walks the lowered TIR op-by-op (:func:`tilelang.tpu.ppl_runner.emit_pl` ->
    :mod:`tilelang.tpu.ppl_codegen`), builds the ``.pl`` with the PPL toolchain,
    and returns a callable :class:`TPUKernel`.  Shapes are baked from the TIR
    (one compile per shape); the emitted ``__KERNEL__`` takes only pointers.

    Compilation is device-agnostic — the device ID is resolved at runtime from
    ``$TPU_VISIBLE_DEVICES`` when the kernel is first called.
    """
    logger.warning("[TPU]: tpu_compiler()")
    if target != "tpu":
        raise ValueError(f"tilelang.tpu.compile only supports target='tpu', got {target!r}.")
    # Accept either a PrimFunc or a callable producing one.
    if callable(func) and not hasattr(func, "buffer_map"):
        func = func()

    info = emit_pl(func, out_idx=out_idx)

    if workdir is None:
        tag = hashlib.md5(info.source.encode()).hexdigest()[:12]
        workdir = os.path.join("/tmp", f"tilelang_tpu_{info.kernel_name}_{tag}")
    paths = build_generic(info, workdir, **build_kw)
    return TPUKernel(info, paths)


# --------------------------------------------------------------------------- #
# Legacy hand-written GEMM template path (compile_gemm / run_profiling)         #
# --------------------------------------------------------------------------- #


class TPUGemmKernel:
    """A compiled TPU GEMM(+ReLU) kernel from the legacy hand-written template.

    Callable as ``c = kernel(a, b)`` with fp16 CPU torch tensors.  Used only by
    :func:`compile_gemm` (runtime-int-shape / multi-core / multi-buffer GEMM).
    """

    def __init__(self, spec: PPLGemmSpec, paths: dict[str, str], *, device: int | None = None):
        self.spec = spec
        self.paths = paths
        self._device = device
        self._runtime: PPLKernel | None = None
        self._a: torch.Tensor | None = None
        self._b: torch.Tensor | None = None

    @property
    def device(self) -> int:
        if self._device is None:
            self._device = get_tpu_device()
        return self._device

    def _ensure_runtime(self) -> PPLKernel:
        if self._runtime is None:
            rt = PPLKernel(self.paths, device=self.device, kernel_name=self.spec.kernel_name)
            rt.init()
            self._runtime = rt
        return self._runtime

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.detach().cpu().contiguous()
        b = b.detach().cpu().contiguous()
        self._a, self._b = a, b
        try:
            return self._ensure_runtime().run(a, b)
        except (RuntimeError, OSError):
            self.close()
            return self._ensure_runtime().run(a, b)

    def get_kernel_source(self, kernel_only: bool = True) -> str:
        from tilelang.tpu.ppl_runner import _emit_gemm_template
        return _emit_gemm_template(self.spec)

    def get_profiler(self, **_kw: Any) -> "_TPUProfiler":
        prof = _TPUProfiler(self)  # type: ignore[arg-type]
        return prof

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _concrete_or_default(val: Any, default: int = 1024) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _shape_tag(val: Any) -> str:
    try:
        return str(int(val))
    except (TypeError, ValueError):
        return "dyn"


def compile_gemm(
    M: int | Any,
    K: int | Any,
    N: int | Any,
    *,
    block_m: int = 64,
    block_k: int = 64,
    block_n: int = 64,
    relu: bool = True,
    in_dtype: str = "fp16",
    core_num: int = 1,
    num_stages: int = 1,
    workdir: str | None = None,
    kernel_name: str = "tl_gemm_relu",
    **build_kw: Any,
) -> TPUGemmKernel:
    """Compile a GEMM(+ReLU) kernel via the legacy hand-written PPL template.

    The PPL ``__KERNEL__`` takes M/K/N as runtime int args, so a single build
    handles any divisible shape; symbolic dims are replaced by defaults (1024)
    only for the ``__TEST__`` stub.  ``core_num`` selects the single/multi-core
    template; ``num_stages`` selects the single/double-buffer template.
    """
    logger.warning("[TPU]: compile_gemm()")
    build_M = _concrete_or_default(M)
    build_K = _concrete_or_default(K)
    build_N = _concrete_or_default(N)
    spec = PPLGemmSpec(
        M=build_M, K=build_K, N=build_N,
        block_m=block_m, block_k=block_k, block_n=block_n,
        relu=relu, in_dtype=in_dtype, core_num=core_num,
        num_stages=num_stages, kernel_name=kernel_name,
    )
    if workdir is None:
        tag_m, tag_k, tag_n = _shape_tag(M), _shape_tag(K), _shape_tag(N)
        mc_tag = f"_mc{core_num}" if core_num > 1 else ""
        nb_tag = f"_nb{num_stages}" if num_stages > 1 else ""
        workdir = os.path.join(
            "/tmp",
            f"tilelang_tpu_{kernel_name}_{tag_m}_{tag_k}_{tag_n}{mc_tag}{nb_tag}",
        )
    paths = build(spec, workdir, **build_kw)
    return TPUGemmKernel(spec, paths)
