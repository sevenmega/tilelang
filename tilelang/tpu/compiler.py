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
    PPLKernel,
    PPLGenericKernel,
    PPLGemmSpec,
    PPLKernelInfo,
    WRAPPER_ABI,
    build,
    build_generic,
    emit_pl,
    get_tpu_device,
    get_torch_device,
    _collect_profile_data,
    _concrete_dim,
    _TORCH_DTYPE,
)
from tilelang.transform.lower_tpu_pipeline import lower_tpu_pipeline
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
        # Static launch plumbing: which tensor extents feed which symbolic dim.
        # See ``PPLKernelInfo.dyn_slots`` -- deriving it per launch was a third
        # of the host-side cost of a kernel launch.
        slots = info.dyn_slots
        self._input_dyn_slots = [s for a, s in zip(info.args, slots) if not a.is_output]
        self._output_dyn_slots = [s for a, s in zip(info.args, slots) if a.is_output]

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

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Launch the kernel with on-device tensors (the only supported ABI).

        ``inputs`` are device tensors, one per non-output PrimFunc param in
        order; their ``.data_ptr()`` values are passed straight to the kernel as
        raw device addresses.  Any output the caller does not supply is
        allocated on the device, and the result is returned as a device tensor.
        Nothing is copied through host memory.

        The optional trailing form ``kernel(*inputs, *outputs)`` writes into
        caller-owned output tensors — the caller's storage is used directly, so
        there is no second allocation and no ``copy_``.  That form is what the
        adapter's write-into mode maps onto.

        CPU tensors are rejected: the host-marshalling path was removed so that
        a caller cannot silently get a slow, copying launch.  Move inputs to the
        device (``t.to(f"tpu:{device}")``) instead.
        """
        in_args = self.info.inputs
        n_in = len(in_args)
        n_out = len(self.info.outputs)
        if not (n_in <= len(inputs) <= n_in + n_out):
            raise ValueError(
                f"{self.info.kernel_name} expects {n_in} inputs (plus up to "
                f"{n_out} preallocated outputs), got {len(inputs)}"
            )
        device_inputs = list(inputs[:n_in])
        provided_outputs = list(inputs[n_in:])

        for t in device_inputs:
            if t.device.type != "tpu":
                raise ValueError(
                    f"{self.info.kernel_name}: inputs must be TPU device "
                    f"tensors (the host-marshalling path was removed); got a "
                    f"tensor on '{t.device.type}'"
                )

        self._last_inputs = device_inputs
        try:
            return self._launch_device(device_inputs, provided_outputs)
        except (RuntimeError, OSError):
            # Device handle may be stale (e.g. invalidated by another config's
            # py_init_device during autotuning).  Reinitialize and retry once.
            self.close()
            return self._launch_device(device_inputs, provided_outputs)

    def _resolve_dims(
        self,
        inputs: list[torch.Tensor],
        provided_outputs: list[torch.Tensor],
    ) -> dict[str, int]:
        """Dynamic dims from the tensors actually passed, in one pass per slot.

        An output-only dim (a destination cache whose extent nothing else
        mentions) is legal as long as the destination is caller-owned; only a
        dim that would have to be guessed to *allocate* a missing output stays
        an error.  ``PPLKernelInfo.dyn_slots`` is what keeps this off the
        stringify path.
        """
        info = self.info
        values: list[int | None] = [None] * len(info.dyn_dims)
        for t, slots in zip(inputs, self._input_dyn_slots):
            for pos, k in slots:
                val = int(t.shape[pos])
                if values[k] is not None and values[k] != val:
                    raise ValueError(
                        f"dynamic dim {info.dyn_dims[k]!r} has conflicting values "
                        f"{values[k]} and {val} across device inputs"
                    )
                values[k] = val
        for t, slots in zip(provided_outputs, self._output_dyn_slots):
            for pos, k in slots:
                if values[k] is None:
                    values[k] = int(t.shape[pos])
        missing = [n for n, v in zip(info.dyn_dims, values) if v is None]
        if missing:
            raise ValueError(
                f"unresolved dynamic dim(s) {missing}; check input tensor shapes"
            )
        return dict(zip(info.dyn_dims, values))

    def _launch_device(
        self,
        inputs: list[torch.Tensor],
        provided_outputs: list[torch.Tensor],
    ) -> torch.Tensor:
        """Resolve dims, allocate any missing outputs on-device, then launch."""
        rt = self._ensure_runtime()
        info = self.info
        dims_by_name = self._resolve_dims(inputs, provided_outputs)

        # Use caller-supplied outputs where given; allocate the rest on the TPU
        # with torch_tpu's allocator (same VA space as the inputs, so the kernel
        # writes directly into this storage).  This is torch_tpu's *logical*
        # device id -- `self.device` is the PPL runtime's physical id, and
        # torch_tpu maps one to the other.
        tpu_device: str | None = None
        out_tensors: list[torch.Tensor] = []
        for i, arg in enumerate(info.outputs):
            if i < len(provided_outputs):
                out = provided_outputs[i]
                if out.device.type != "tpu":
                    raise ValueError(
                        f"{info.kernel_name}: preallocated output {i} must be a "
                        f"TPU device tensor; got '{out.device.type}'"
                    )
                out_tensors.append(out)
            else:
                if tpu_device is None:
                    tpu_device = f"tpu:{get_torch_device()}"
                shape = tuple(_concrete_dim(s, dims_by_name) for s in arg.shape)
                dtype = _TORCH_DTYPE[arg.torch_dtype]
                out_tensors.append(
                    torch.empty(shape, dtype=dtype, device=tpu_device)
                )

        input_addrs = [int(t.data_ptr()) for t in inputs]
        output_addrs = [int(t.data_ptr()) for t in out_tensors]
        rt.run_device_ptrs(input_addrs, dims_by_name, output_addrs)
        # run_device_ptrs re-asserts the runtime's current device immediately
        # before the launch, and leaves the kernel ordered against the caller's
        # torch stream, so neither a device re-sync nor a fence is needed here.

        return out_tensors[0] if len(out_tensors) == 1 else tuple(out_tensors)

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
            # Pinned here so collection can pick this device's dump out of the
            # dir even if TPU_VISIBLE_DEVICES changes before collect_profile().
            "device": self.device,
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
        return _collect_profile_data(
            self._profile_config["dir"],
            device=self._profile_config.get("device", self.device),
            verbose=verbose,
        )

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
    and returns a callable :class:`TPUKernel`.

    Shapes may be **baked** (concrete extents in the TIR — one build per shape) or
    **dynamic**: any buffer dimension declared with ``T.dynamic`` stays symbolic in
    ``buffer_map``, is emitted as a trailing ``int`` runtime argument of the PPL
    ``__KERNEL__``, and is resolved per call from the input tensors' shapes. On-chip
    tile extents must always be concrete (``emit_local_allocs`` bakes them to ints),
    so tile sizes / head-dim style dims stay baked while batch / sequence / head
    counts can be dynamic. A single dynamic build then serves every shape whose
    tiles divide evenly.

    Compilation is device-agnostic — the device ID is resolved at runtime from
    ``$TPU_VISIBLE_DEVICES`` when the kernel is first called.
    """
    logger.warning("[TPU]: tpu_compiler()")
    if target != "tpu":
        raise ValueError(f"tilelang.tpu.compile only supports target='tpu', got {target!r}.")
    # Accept either a PrimFunc or a callable producing one.
    if callable(func) and not hasattr(func, "buffer_map"):
        func = func()

    # Real num_stages pipelining: rewrite the T.Pipelined loop into an explicit
    # prologue / S-way-unrolled steady state / drain, multi-buffering the
    # pipeline-stage locals so every buffer reference is a compile-time
    # constant.  Must run before emit_pl; a no-op when the kernel has no loop
    # tagged ``num_stages >= 2``.
    #
    # TILELANG_TPU_PIPELINE=0 restores the old behaviour (tag the loop and let
    # PPL's implicit scheduler handle it), which is what the pass is measured
    # against.
    if os.environ.get("TILELANG_TPU_PIPELINE", "1") != "0":
        func = lower_tpu_pipeline(func)

    info = emit_pl(func, out_idx=out_idx)

    if workdir is None:
        # ``WRAPPER_ABI`` is mixed in so a change to the ctypes wrapper's
        # build layout (link flags, translation unit) invalidates the cached
        # workdir -- the device-side ``.pl`` is unchanged by such a change.
        tag = hashlib.md5(
            f"{WRAPPER_ABI}\0{info.source}".encode()
        ).hexdigest()[:12]
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
        """Launch with device tensors (the only supported ABI).

        The legacy host marshalling (``.cpu()`` + H2D) was removed together with
        the generic path's; both now hand raw device addresses to the kernel.
        """
        for t in (a, b):
            if t.device.type != "tpu":
                raise ValueError(
                    f"{self.spec.kernel_name}: inputs must be TPU device "
                    f"tensors (the host-marshalling path was removed); got a "
                    f"tensor on '{t.device.type}'"
                )
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
            f"tilelang_tpu_{kernel_name}_{tag_m}_{tag_k}_{tag_n}{mc_tag}{nb_tag}"
            f"_{WRAPPER_ABI}",
        )
    paths = build(spec, workdir, **build_kw)
    return TPUGemmKernel(spec, paths)
