"""PPL build & run engine for tilelang's TPU backend.

This module packages the *verified* recipe for running a tilelang-style GEMM
kernel on SG2260E TPU hardware via the PPL toolchain.  The recipe mirrors the
canonical reference kernel ``PPLOPs/pplops/ops/gemm_w8a8bf16.pl``:

  1. emit a PPL ``.pl`` kernel (``__KERNEL__`` + ``__TEST__``);
  2. ``ppl_compile.py --gen_test`` builds ``lib/libkernel.so`` + a host launcher
     (``host/*.cpp``, ``include/*.h``) + a CMakeLists.txt;
  3. append a small ctypes wrapper target (``tl_py.so``) that exposes
     ``py_init_device / py_init_device_shared / py_sync_device / py_flush /
     py_dev_malloc / py_memcpy_h2d / py_memcpy_d2h / py_<kernel>`` and rebuild
     that target;
  4. from Python, ``ctypes``-load ``tl_py.so``, copy torch tensors H2D, launch,
     sync, copy D2H.

The fp16 GEMM+ReLU kernel emitted here (M=N=K=1024, 64x64x64 tiles, fp32 accum)
was compiled and run on real TPU ``devid 2`` and matched ``torch.relu(a @ b)``
within ``rtol=atol=1e-2`` (max diff 0.015625) -- i.e. it passes the assertion in
``test_tilelang/test_gemm_naive.py``.

Only the fp16 -> fp32-accum -> (relu) -> fp16 path is verified end-to-end today.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import logging

from tilelang.tpu.ppl_codegen import translate, PPLKernelInfo, BufArg  # noqa: F401

logger = logging.getLogger(__name__)

_TORCH_DTYPE = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

# ctypes handle to tpuRtGetDevice, resolved lazily. None = not yet tried,
# False = tried and unavailable (don't retry).
_TPURT_GET_DEVICE: Any = None
# ctypes handle to tpuRtSetDevice, resolved lazily. None = not yet tried,
# False = tried and unavailable (don't retry).
_TPURT_SET_DEVICE: Any = None


def _reassert_tpu_device(device: int) -> None:
    """Re-point the PPL runtime's current device at ``device`` before a launch.

    Any torch_tpu device-tensor op (e.g. allocating the kernel's output with
    ``torch.empty(..., device='tpu')``) silently moves the runtime's notion of
    the current device, after which a kernel launch dies with
    "stream and device mismatch!".  The device-pointer launch path allocates its
    output *after* the caller's sync, so it must re-assert here, immediately
    before the launch.  A no-op if the runtime symbol is unavailable.
    """
    global _TPURT_SET_DEVICE
    if _TPURT_SET_DEVICE is None:
        try:
            # The runtime is already mapped by torch_tpu; CDLL(None) searches the
            # global symbol namespace without re-opening the .so by path.
            lib = ctypes.CDLL(None, mode=ctypes.RTLD_GLOBAL)
            fn = lib.tpuRtSetDevice
            fn.argtypes = [ctypes.c_int]
            fn.restype = ctypes.c_int
            _TPURT_SET_DEVICE = fn
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[TPU] cannot re-assert the PPL runtime device (%s); "
                "device-pointer launches may fail after a torch_tpu op.",
                exc,
            )
            _TPURT_SET_DEVICE = False
    if _TPURT_SET_DEVICE:
        _TPURT_SET_DEVICE(int(device))


#: ``PPL_ASYNC_LAUNCH=1`` selects the flush+event launch (see
#: ``PPLGenericKernel._ordered_launch``).  OFF by default: with a *per-wrapper*
#: stream it faulted a TPU chip in a model run (see that method's docstring).
#: It is only safe together with :func:`_ppl_shared_stream`, which every kernel
#: on the device now shares.
_ASYNC_LAUNCH = os.environ.get("PPL_ASYNC_LAUNCH", "") not in ("", "0")

#: One torch-owned stream per device, shared by every tilelang kernel on it:
#: ``device -> (torch.tpu.Stream, raw tpuStream_t)``.
_PPL_STREAMS: dict[int, tuple[Any, int]] = {}


def _ppl_shared_stream(device: int) -> tuple[Any, int] | None:
    """The one PPL stream every tilelang kernel on ``device`` launches into.

    A wrapper that creates its own stream is invisible to every other wrapper,
    so two kernels' launches cannot be ordered against each other, and flushing
    them out of order is what faulted a chip.  Sharing one stream makes the
    ordering total: a single flush publishes onto the same stream for all of
    them, exactly as torch_tpu runs (one stream per device, one module per
    kernel).

    The stream is created by *torch* (``torch.tpu.Stream``), not by
    ``tpuRtStreamCreate``, so the runtime already knows it: events recorded on
    it are ordinary stream events (no ``ExternalStream`` wrapper) and the
    device-wide sync path walks it like any other torch stream.  Returns
    ``(stream, raw_ptr)``, or ``None`` when torch exposes no ``Stream``, which
    puts the caller back on a private stream.
    """
    cached = _PPL_STREAMS.get(device)
    if cached is not None:
        return cached
    try:
        _reassert_tpu_device(device)
        stream = torch.tpu.Stream(device=get_torch_device())
        ptr = int(stream._as_parameter_.value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[TPU] no shared PPL stream (%s); using a private one", exc)
        return None
    _PPL_STREAMS[device] = (stream, ptr)
    return stream, ptr


def _sync_torch_stream() -> None:
    """Order the caller's torch_tpu writes before the PPL launch.

    ``torch_tpu`` enqueues its ops (``randn``, ``ones``, ``fill_``, slice
    assignments, ``copy_``) on *its own* stream.  The PPL wrapper launches on
    the PPL stream (shared across kernels, or private to one wrapper when the
    async launch is off).  Nothing orders the two,
    so without a sync the kernel can read input buffers while the writes that
    filled them are still in flight -- the kernel then computes on uninitialized
    (typically zero) memory and the caller sees zeros, denormals, or NaN.

    Syncs torch_tpu's own current (logical) device: that is where the buffers
    actually live, and it is not the PPL runtime's physical id.  A no-op when
    torch_tpu is not in use, so host-only tooling is unaffected.
    """
    try:
        import torch

        torch.get_device_module("tpu").synchronize()
    except Exception as exc:  # noqa: BLE001
        logger.debug("[TPU] cannot sync the torch_tpu stream (%s)", exc)



# --------------------------------------------------------------------------- #
# PPL root resolution                                                          #
# --------------------------------------------------------------------------- #

def get_torch_device() -> int:
    """Return torch_tpu's *logical* device index (default 0).

    This is the index tensors are allocated on (``device="tpu:N"``) and the one
    ``torch.tpu.set_device`` selects.  It is **not** the index the PPL runtime
    wants: torch_tpu maps logical ids onto a different set of physical chips
    (e.g. logical 7 -> physical 4), see :func:`get_tpu_device`.
    """
    try:
        import torch

        mod = getattr(torch, "tpu", None) or torch.get_device_module("tpu")
        return int(mod.current_device())
    except Exception:  # noqa: BLE001  (torch_tpu not imported / not available)
        pass
    val = os.environ.get("TPU_VISIBLE_DEVICES", "0")
    try:
        return int(val.split(",")[0])
    except (ValueError, IndexError):
        return 0


def get_tpu_device() -> int:
    """Return the **physical** device index the PPL runtime should bind to.

    ``torch_tpu`` and the PPL runtime disagree about device numbering:
    ``torch.tpu.set_device(7)`` binds torch to logical 7, but the runtime's own
    ``tpuGetDevice`` reports 4 -- torch_tpu maps logical ids onto physical chips
    through its chip table.  Handing the logical id to ``tpuRtSetDevice`` bound
    the wrapper's stream to the wrong chip, so a launch with a ``tpu:7`` tensor
    died with "stream and device mismatch" and the output stayed zero.

    So this returns the runtime's *physical* id, read after torch_tpu has been
    initialized (its first allocation or ``set_device`` publishes the mapping).
    Falls back to ``$TPU_VISIBLE_DEVICES`` (default 0) without torch_tpu.
    """
    try:
        import torch  # noqa: F401
        import torch_tpu  # noqa: F401  (maps the runtime and its symbols)

        torch.tpu.current_device()  # force lazy init: publishes the chip map
        fn = _tpu_rt_get_device()
        if fn is not None:
            dev = ctypes.c_int()
            if fn(ctypes.byref(dev)) == 0:
                return int(dev.value)
    except Exception:  # noqa: BLE001
        pass
    val = os.environ.get("TPU_VISIBLE_DEVICES", "0")
    try:
        return int(val.split(",")[0])
    except (ValueError, IndexError):
        return 0


def _tpu_rt_get_device():
    """Resolve ``tpuRtGetDevice`` from the already-mapped runtime, lazily."""
    global _TPURT_GET_DEVICE
    if _TPURT_GET_DEVICE is None:
        try:
            lib = ctypes.CDLL(None, mode=ctypes.RTLD_GLOBAL)
            fn = lib.tpuRtGetDevice
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            _TPURT_GET_DEVICE = fn
        except Exception as exc:  # noqa: BLE001
            logger.debug("[TPU] no tpuRtGetDevice (%s); using the env device", exc)
            _TPURT_GET_DEVICE = False
    return _TPURT_GET_DEVICE or None


def ensure_tpu_device() -> int:
    """Register the ``"tpu"`` torch device, then return its *logical* index.

    ``torch.randn(..., device="tpu:0")`` raises "Expected one of cpu, cuda,
    ..." until ``torch_tpu`` has been imported -- that import is what
    registers the privateuseone backend under the name ``tpu``.  Tests and
    scripts that build device tensors before touching a kernel must call this
    first; kernel calls themselves work without it only because they allocate
    outputs after the runtime is already initialized.

    The value is the logical id for ``device="tpu:N"``.  Kernels resolve their
    own (physical) launch device internally -- see :func:`get_tpu_device`.
    """
    import torch_tpu  # noqa: F401  (registers the "tpu" device type)

    return get_torch_device()


def _get_ppl_root() -> str:
    """Return the path to the PPL release package via $PPL_PROJECT_ROOT.

    The user must source the PPL ``envsetup.sh`` (or set ``PPL_PROJECT_ROOT``
    manually) before using the TPU backend.
    """
    root = os.environ.get("PPL_PROJECT_ROOT", "")
    if root and os.path.isdir(root):
        return root
    raise RuntimeError(
        "PPL_PROJECT_ROOT is not set or does not point to a valid directory. "
        "Source the PPL envsetup.sh first "
        "(e.g. `source /path/to/ppl_v1.7.198-.../envsetup.sh`)."
    )


_CHIP_MAP_CACHE: dict[str, str] | None = None


def _resolve_chip_arch(ppl_root: str, chip: str) -> str:
    """Map user-facing chip name (e.g. 'sg2260e') to arch code ('tpub_7_1_e')."""
    global _CHIP_MAP_CACHE
    if _CHIP_MAP_CACHE is None:
        map_file = os.path.join(ppl_root, "deps", "chip", "chip_map.json")
        with open(map_file) as f:
            _CHIP_MAP_CACHE = json.load(f)
    if chip in _CHIP_MAP_CACHE:
        return _CHIP_MAP_CACHE[chip]
    if chip in _CHIP_MAP_CACHE.values():
        return chip
    raise ValueError(f"Unknown chip '{chip}'; known: {list(_CHIP_MAP_CACHE.keys())}")


@contextlib.contextmanager
def _scoped_chip_env(keys: tuple[str, ...]):
    """Snapshot *keys* and restore them exactly — including unset — on exit.

    The rest of ``_setup_ppl_env`` mutates the host process env so the
    ``ppl-compile`` / ``cmake`` **subprocesses** inherit it, but ``CHIP`` /
    ``CHIP_ARCH`` are also read *in-process* by the device runtime:
    ``ppl_get_arch_info()`` (ppltensor.h) does a strcmp over arch NAMES and
    treats ``CHIP_ARCH`` as authoritative, falling back to ``CHIP``.  The build
    needs the chip CODE (``tpub_7_1_e``) there, the runtime needs the arch NAME
    (``sg2260e``).  Writing the code in and left in place poisons every
    subsequent device op — an in-forward kernel build therefore made the next
    op (``o_proj``) throw ``Unsupported CHIP: tpub_7_1_e``.  Both subprocess
    paths already receive the code explicitly (``ppl-compile --chip``,
    ``cmake -DCHIP``), so the mutation only has to live as far as process
    spawn: scoping it here stops the leak at its source.

    ``clear()`` (not ``set(default)``) is used when a key was originally
    unset, so a faithful restore never fabricates a value the caller must not
    see — ``ppl_get_arch_info()`` rejects an empty string outright.
    """
    saved = {k: (k in os.environ, os.environ.get(k)) for k in keys}
    try:
        yield
    finally:
        for k, (was_set, value) in saved.items():
            if was_set:
                os.environ[k] = value
            else:
                os.environ.pop(k, None)


def _setup_ppl_env(
    ppl_root: str,
    chip: str,
    chip_arch: str,
    workdir: str,
    kernel_name: str,
    mode: str = "pcie",
) -> None:
    """Set env vars that ppl-compile and the CMake build expect.

    Device ID is intentionally absent — compilation is device-agnostic.
    ``PPL_DEVID`` is set at runtime by ``_ensure_tpu_env()`` / ``run_profiling()``.
    """
    deps = os.path.join(ppl_root, "deps")
    os.environ["PPL_PROJECT_ROOT"] = ppl_root
    os.environ["PPL_RUNTIME_PATH"] = deps
    os.environ["PPL_THIRD_PARTY_PATH"] = os.path.join(ppl_root, "third_party")
    os.environ["CROSS_TOOLCHAINS"] = os.path.join(ppl_root, "third_party", "toolchains_dir")
    os.environ["CHIP"] = chip
    os.environ["CHIP_ARCH"] = chip_arch
    os.environ["PPL_TPUKERNEL_DEV_MODE"] = mode
    os.environ["PPL_CACHE_PATH"] = os.path.join(workdir, "cache")
    os.environ["PPL_FILE_NAME"] = kernel_name
    os.environ["PPL_DATA_PATH"] = os.path.join(workdir, "data")
    os.environ["PPL_SRC_DIR_PATH"] = workdir

    chip_lib = os.path.join(deps, "chip", chip_arch, "lib")
    rt_lib = os.path.join(deps, "runtime", "tpuv7-runtime", "lib")
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_set = set(ld.split(":")) if ld else set()
    if mode == "pcie":
        # Real driver (/opt/tpuv7/) must precede the emulator copies in deps/runtime/
        tpuv7_lib = "/opt/tpuv7/tpuv7-current/lib"
        parts = [os.path.join(workdir, "lib"), tpuv7_lib, chip_lib, rt_lib]
    else:
        parts = [os.path.join(workdir, "lib"), chip_lib, rt_lib]
    new = [p for p in parts if p not in ld_set]
    if new:
        prefix = ":".join(new)
        ld = prefix + ":" + ld if ld else prefix
    os.environ["LD_LIBRARY_PATH"] = ld

    if mode == "pcie":
        os.environ["PPL_KERNEL_PATH"] = os.path.join(workdir, "lib", "libkernel.so")


def _run_ppl_compile(
    ppl_root: str,
    pl_path: str,
    chip_arch: str,
    workdir: str,
    opt: str = "O3",
    rv: bool = True,
    autotune: bool = False,
    verbose: bool = False,
) -> None:
    """Run the ppl-compile MLIR binary to generate device/host code.

    When *autotune* is True, ``--autotune`` is passed instead of ``--gen-test``
    so the generated code includes profiling instrumentation.
    """
    compiler = os.path.join(ppl_root, "bin", "ppl-compile")
    cmd = [
        compiler,
        pl_path,
        "--print-debug-info",
        "--print-ir",
        "--chip", chip_arch,
        f"--{opt}",
        "--g",
        "--o", workdir,
        "--autotune" if autotune else "--gen-test",
    ]
    if rv:
        cmd.append("--rv")
    if verbose:
        print("[ppl_runner] ppl-compile:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _cmake_build(
    ppl_root: str,
    chip_arch: str,
    workdir: str,
    mode: str = "pcie",
    verbose: bool = False,
) -> None:
    """Copy pcie.cmake as CMakeLists.txt and run cmake + make."""
    cmake_template = os.path.join(ppl_root, "deps", "scripts", f"{mode}.cmake")
    shutil.copy(cmake_template, os.path.join(workdir, "CMakeLists.txt"))

    build_dir = os.path.join(workdir, "build")
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = (
        f"cmake .. -DDEBUG=False -DCHIP={chip_arch} -DDEV_MODE={mode}"
        f" -DEXTRA_IDIRS= -DEXTRA_LDIRS= -DEXTRA_CFLAGS= -DEXTRA_LDFLAGS= -DUSE_MPI=False"
    )
    make_cmd = "make install"
    if verbose:
        make_cmd = "make install VERBOSE=1"
    for cmd in (cmake_cmd, make_cmd):
        if verbose:
            print(f"[ppl_runner] {cmd}")
        subprocess.run(cmd, shell=True, check=True, cwd=build_dir)

# --------------------------------------------------------------------------- #
# PPL kernel source template                                                   #
# --------------------------------------------------------------------------- #

# fp16/bf16 GEMM (+ optional ReLU) with fp32 accumulator, 4D [1,M,1,N] layout.
# Tailored to tile sizes that divide M/K/N evenly (no boundary handling),
# exactly like the hardware-verified /tmp/tl_gemm_relu.pl.
# {in_type} is "fp16" or "bf16" (a wild-guess bf16 path per the TPU bring-up plan;
# correctness is not required for bf16, only lowering/build/run).
_PL_TEMPLATE = r"""#include "ppl.h"

using namespace ppl;

// C[M,N] = (relu?)(A[M,K] @ B[K,N]);  A/B {in_type}, accum fp32, C {in_type}.
// Tiles block_m x block_k x block_n must divide M/K/N evenly.
__KERNEL__ void {kernel_name}({in_type} *ptr_res, {in_type} *ptr_left, {in_type} *ptr_right,
                              int M, int K, int N) {{
  const int block_m = {block_m};
  const int block_k = {block_k};
  const int block_n = {block_n};

  dim4 res_global_shape   = {{1, M, 1, N}};
  dim4 left_global_shape  = {{1, M, 1, K}};
  dim4 right_global_shape = {{1, K, 1, N}};

  auto res_gtensor   = gtensor<{in_type}>(res_global_shape,   GLOBAL, ptr_res);
  auto left_gtensor  = gtensor<{in_type}>(left_global_shape,  GLOBAL, ptr_left);
  auto right_gtensor = gtensor<{in_type}>(right_global_shape, GLOBAL, ptr_right);

  dim4 res_max_shape   = {{1, block_m, 1, block_n}};
  dim4 left_max_shape  = {{1, block_m, 1, block_k}};
  dim4 right_max_shape = {{1, block_k, 1, block_n}};

  auto sub_left  = tensor<{in_type}>(left_max_shape);
  auto sub_right = tensor<{in_type}>(right_max_shape);
  auto res_{in_type}  = tensor<{in_type}>(res_max_shape, TPU_COMPACT);

  for (int idx_m = 0; idx_m < M; idx_m += block_m) {{
    for (int idx_n = 0; idx_n < N; idx_n += block_n) {{
      auto sub_res = make_tensor<fp32>(res_max_shape, res_max_shape);
      tiu::zero(sub_res);
      for (int idx_k = 0; idx_k < K; idx_k += block_k) {{
        enable_pipeline();
        dim4 left_offset  = {{0, idx_m, 0, idx_k}};
        dim4 right_offset = {{0, idx_k, 0, idx_n}};
        dma::load(sub_left,
                  left_gtensor.sub_view(left_max_shape, left_offset));
        dma::load(sub_right,
                  right_gtensor.sub_view(right_max_shape, right_offset));
        bool last_k = (K - idx_k <= block_k);
        int bias = 0;
        bool saturate = false;
        float requant = 1;
        tiu::fmm2_nn(sub_res, sub_left, sub_right, bias, /*result_add=*/true,
                     DT_FP32, /*do_relu=*/({do_relu_int} && last_k), saturate, requant);
      }}
      tiu::cast(res_{in_type}, sub_res);
      dim4 res_offset = {{0, idx_m, 0, idx_n}};
      dma::store(res_gtensor.sub_view(res_max_shape, res_offset), res_{in_type});
    }}
  }}
}}

__TEST__ void {kernel_name}_main() {{
  const int M = {M};
  const int K = {K};
  const int N = {N};
  dim4 res_shape   = {{1, M, 1, N}};
  dim4 left_shape  = {{1, M, 1, K}};
  dim4 right_shape = {{1, K, 1, N}};
  {in_type} *res   = malloc<{in_type}>(&res_shape);
  rand(res, &res_shape, -1.0, 1.0);
  {in_type} *left  = malloc<{in_type}>(&left_shape);
  rand(left, &left_shape, -1.0, 1.0);
  {in_type} *right = malloc<{in_type}>(&right_shape);
  rand(right, &right_shape, -1.0, 1.0);
  {kernel_name}(res, left, right, M, K, N);
}}
"""

# Multi-core variant: partitions M across cores using set_block_num_max().
# Same format placeholders as _PL_TEMPLATE.
_PL_TEMPLATE_MULTICORE = r"""#include "ppl.h"

using namespace ppl;

// Multi-core C[M,N] = (relu?)(A[M,K] @ B[K,N]); M partitioned across cores.
__KERNEL__ void {kernel_name}({in_type} *ptr_res, {in_type} *ptr_left, {in_type} *ptr_right,
                              int M, int K, int N) {{
  set_block_num_max();
  int core_num = get_block_num();
  int core_idx = get_block_index();
  if (core_idx >= core_num) return;

  const int block_m = {block_m};
  const int block_k = {block_k};
  const int block_n = {block_n};

  int M_tiles = (M + block_m - 1) / block_m;
  int tiles_per_core = (M_tiles + core_num - 1) / core_num;
  int m_start = core_idx * tiles_per_core * block_m;
  int m_end   = min(m_start + tiles_per_core * block_m, M);

  dim4 res_global_shape   = {{1, M, 1, N}};
  dim4 left_global_shape  = {{1, M, 1, K}};
  dim4 right_global_shape = {{1, K, 1, N}};

  auto res_gtensor   = gtensor<{in_type}>(res_global_shape,   GLOBAL, ptr_res);
  auto left_gtensor  = gtensor<{in_type}>(left_global_shape,  GLOBAL, ptr_left);
  auto right_gtensor = gtensor<{in_type}>(right_global_shape, GLOBAL, ptr_right);

  dim4 res_max_shape   = {{1, block_m, 1, block_n}};
  dim4 left_max_shape  = {{1, block_m, 1, block_k}};
  dim4 right_max_shape = {{1, block_k, 1, block_n}};

  auto sub_left  = tensor<{in_type}>(left_max_shape);
  auto sub_right = tensor<{in_type}>(right_max_shape);
  auto res_{in_type}  = tensor<{in_type}>(res_max_shape, TPU_COMPACT);

  for (int idx_m = m_start; idx_m < m_end; idx_m += block_m) {{
    for (int idx_n = 0; idx_n < N; idx_n += block_n) {{
      auto sub_res = make_tensor<fp32>(res_max_shape, res_max_shape);
      tiu::zero(sub_res);
      for (int idx_k = 0; idx_k < K; idx_k += block_k) {{
        enable_pipeline();
        dim4 left_offset  = {{0, idx_m, 0, idx_k}};
        dim4 right_offset = {{0, idx_k, 0, idx_n}};
        dma::load(sub_left,
                  left_gtensor.sub_view(left_max_shape, left_offset));
        dma::load(sub_right,
                  right_gtensor.sub_view(right_max_shape, right_offset));
        bool last_k = (K - idx_k <= block_k);
        int bias = 0;
        bool saturate = false;
        float requant = 1;
        tiu::fmm2_nn(sub_res, sub_left, sub_right, bias, /*result_add=*/true,
                     DT_FP32, /*do_relu=*/({do_relu_int} && last_k), saturate, requant);
      }}
      tiu::cast(res_{in_type}, sub_res);
      dim4 res_offset = {{0, idx_m, 0, idx_n}};
      dma::store(res_gtensor.sub_view(res_max_shape, res_offset), res_{in_type});
    }}
  }}
}}

__TEST__ void {kernel_name}_main() {{
  const int M = {M};
  const int K = {K};
  const int N = {N};
  dim4 res_shape   = {{1, M, 1, N}};
  dim4 left_shape  = {{1, M, 1, K}};
  dim4 right_shape = {{1, K, 1, N}};
  {in_type} *res   = malloc<{in_type}>(&res_shape);
  rand(res, &res_shape, -1.0, 1.0);
  {in_type} *left  = malloc<{in_type}>(&left_shape);
  rand(left, &left_shape, -1.0, 1.0);
  {in_type} *right = malloc<{in_type}>(&right_shape);
  rand(right, &right_shape, -1.0, 1.0);
  {kernel_name}(res, left, right, M, K, N);
}}
"""


# Single-core double-buffer: explicit ping-pong with parallel_start/parallel_end.
_PL_TEMPLATE_MULTIBUF = r"""#include "ppl.h"

using namespace ppl;

// Double-buffered C[M,N] = (relu?)(A[M,K] @ B[K,N]); explicit ping-pong buffers.
__KERNEL__ void {kernel_name}({in_type} *ptr_res, {in_type} *ptr_left, {in_type} *ptr_right,
                              int M, int K, int N) {{
  const int block_m = {block_m};
  const int block_k = {block_k};
  const int block_n = {block_n};

  dim4 res_global_shape   = {{1, M, 1, N}};
  dim4 left_global_shape  = {{1, M, 1, K}};
  dim4 right_global_shape = {{1, K, 1, N}};

  auto res_gtensor   = gtensor<{in_type}>(res_global_shape,   GLOBAL, ptr_res);
  auto left_gtensor  = gtensor<{in_type}>(left_global_shape,  GLOBAL, ptr_left);
  auto right_gtensor = gtensor<{in_type}>(right_global_shape, GLOBAL, ptr_right);

  dim4 res_max_shape   = {{1, block_m, 1, block_n}};
  dim4 left_max_shape  = {{1, block_m, 1, block_k}};
  dim4 right_max_shape = {{1, block_k, 1, block_n}};

  auto sub_left_0  = tensor<{in_type}>(left_max_shape);
  auto sub_left_1  = tensor<{in_type}>(left_max_shape);
  auto sub_right_0 = tensor<{in_type}>(right_max_shape);
  auto sub_right_1 = tensor<{in_type}>(right_max_shape);
  auto res_{in_type}  = tensor<{in_type}>(res_max_shape, TPU_COMPACT);

  for (int idx_m = 0; idx_m < M; idx_m += block_m) {{
    for (int idx_n = 0; idx_n < N; idx_n += block_n) {{
      auto sub_res = make_tensor<fp32>(res_max_shape, res_max_shape);
      tiu::zero(sub_res);
      int K_iters = K / block_k;
      int bias = 0;
      bool saturate = false;
      float requant = 1;

      // Prologue: load first tile pair into buffer 0
      dma::load(sub_left_0,
                left_gtensor.sub_view(left_max_shape, {{0, idx_m, 0, 0}}));
      dma::load(sub_right_0,
                right_gtensor.sub_view(right_max_shape, {{0, 0, 0, idx_n}}));

      // Main loop: overlap load[i+1] with compute[i]
      int ping = 0;
      for (int ki = 0; ki < K_iters - 1; ki++) {{
        int next_k = (ki + 1) * block_k;
        parallel_start();
        if (ping == 0) {{
          dma::load(sub_left_1,
                    left_gtensor.sub_view(left_max_shape, {{0, idx_m, 0, next_k}}));
          dma::load(sub_right_1,
                    right_gtensor.sub_view(right_max_shape, {{0, next_k, 0, idx_n}}));
          tiu::fmm2_nn(sub_res, sub_left_0, sub_right_0, bias, true, DT_FP32,
                       false, saturate, requant);
        }} else {{
          dma::load(sub_left_0,
                    left_gtensor.sub_view(left_max_shape, {{0, idx_m, 0, next_k}}));
          dma::load(sub_right_0,
                    right_gtensor.sub_view(right_max_shape, {{0, next_k, 0, idx_n}}));
          tiu::fmm2_nn(sub_res, sub_left_1, sub_right_1, bias, true, DT_FP32,
                       false, saturate, requant);
        }}
        parallel_end();
        ping = 1 - ping;
      }}

      // Epilogue: compute last iteration (relu only here)
      if (ping == 0) {{
        tiu::fmm2_nn(sub_res, sub_left_0, sub_right_0, bias, true, DT_FP32,
                     {do_relu_int} != 0, saturate, requant);
      }} else {{
        tiu::fmm2_nn(sub_res, sub_left_1, sub_right_1, bias, true, DT_FP32,
                     {do_relu_int} != 0, saturate, requant);
      }}

      tiu::cast(res_{in_type}, sub_res);
      dim4 res_offset = {{0, idx_m, 0, idx_n}};
      dma::store(res_gtensor.sub_view(res_max_shape, res_offset), res_{in_type});
    }}
  }}
}}

__TEST__ void {kernel_name}_main() {{
  const int M = {M};
  const int K = {K};
  const int N = {N};
  dim4 res_shape   = {{1, M, 1, N}};
  dim4 left_shape  = {{1, M, 1, K}};
  dim4 right_shape = {{1, K, 1, N}};
  {in_type} *res   = malloc<{in_type}>(&res_shape);
  rand(res, &res_shape, -1.0, 1.0);
  {in_type} *left  = malloc<{in_type}>(&left_shape);
  rand(left, &left_shape, -1.0, 1.0);
  {in_type} *right = malloc<{in_type}>(&right_shape);
  rand(right, &right_shape, -1.0, 1.0);
  {kernel_name}(res, left, right, M, K, N);
}}
"""

# Multi-core + double-buffer: M partitioned across cores, explicit ping-pong buffers.
_PL_TEMPLATE_MULTICORE_MULTIBUF = r"""#include "ppl.h"

using namespace ppl;

// Multi-core double-buffered C[M,N] = (relu?)(A[M,K] @ B[K,N]).
// M partitioned across cores; explicit ping-pong with parallel_start/parallel_end.
__KERNEL__ void {kernel_name}({in_type} *ptr_res, {in_type} *ptr_left, {in_type} *ptr_right,
                              int M, int K, int N) {{
  set_block_num_max();
  int core_num = get_block_num();
  int core_idx = get_block_index();
  if (core_idx >= core_num) return;

  const int block_m = {block_m};
  const int block_k = {block_k};
  const int block_n = {block_n};

  int M_tiles = (M + block_m - 1) / block_m;
  int tiles_per_core = (M_tiles + core_num - 1) / core_num;
  int m_start = core_idx * tiles_per_core * block_m;
  int m_end   = min(m_start + tiles_per_core * block_m, M);

  dim4 res_global_shape   = {{1, M, 1, N}};
  dim4 left_global_shape  = {{1, M, 1, K}};
  dim4 right_global_shape = {{1, K, 1, N}};

  auto res_gtensor   = gtensor<{in_type}>(res_global_shape,   GLOBAL, ptr_res);
  auto left_gtensor  = gtensor<{in_type}>(left_global_shape,  GLOBAL, ptr_left);
  auto right_gtensor = gtensor<{in_type}>(right_global_shape, GLOBAL, ptr_right);

  dim4 res_max_shape   = {{1, block_m, 1, block_n}};
  dim4 left_max_shape  = {{1, block_m, 1, block_k}};
  dim4 right_max_shape = {{1, block_k, 1, block_n}};

  auto sub_left_0  = tensor<{in_type}>(left_max_shape);
  auto sub_left_1  = tensor<{in_type}>(left_max_shape);
  auto sub_right_0 = tensor<{in_type}>(right_max_shape);
  auto sub_right_1 = tensor<{in_type}>(right_max_shape);
  auto res_{in_type}  = tensor<{in_type}>(res_max_shape, TPU_COMPACT);

  for (int idx_m = m_start; idx_m < m_end; idx_m += block_m) {{
    for (int idx_n = 0; idx_n < N; idx_n += block_n) {{
      auto sub_res = make_tensor<fp32>(res_max_shape, res_max_shape);
      tiu::zero(sub_res);
      int K_iters = K / block_k;
      int bias = 0;
      bool saturate = false;
      float requant = 1;

      // Prologue: load first tile pair into buffer 0
      dma::load(sub_left_0,
                left_gtensor.sub_view(left_max_shape, {{0, idx_m, 0, 0}}));
      dma::load(sub_right_0,
                right_gtensor.sub_view(right_max_shape, {{0, 0, 0, idx_n}}));

      // Main loop: overlap load[i+1] with compute[i]
      int ping = 0;
      for (int ki = 0; ki < K_iters - 1; ki++) {{
        int next_k = (ki + 1) * block_k;
        parallel_start();
        if (ping == 0) {{
          dma::load(sub_left_1,
                    left_gtensor.sub_view(left_max_shape, {{0, idx_m, 0, next_k}}));
          dma::load(sub_right_1,
                    right_gtensor.sub_view(right_max_shape, {{0, next_k, 0, idx_n}}));
          tiu::fmm2_nn(sub_res, sub_left_0, sub_right_0, bias, true, DT_FP32,
                       false, saturate, requant);
        }} else {{
          dma::load(sub_left_0,
                    left_gtensor.sub_view(left_max_shape, {{0, idx_m, 0, next_k}}));
          dma::load(sub_right_0,
                    right_gtensor.sub_view(right_max_shape, {{0, next_k, 0, idx_n}}));
          tiu::fmm2_nn(sub_res, sub_left_1, sub_right_1, bias, true, DT_FP32,
                       false, saturate, requant);
        }}
        parallel_end();
        ping = 1 - ping;
      }}

      // Epilogue: compute last iteration (relu only here)
      if (ping == 0) {{
        tiu::fmm2_nn(sub_res, sub_left_0, sub_right_0, bias, true, DT_FP32,
                     {do_relu_int} != 0, saturate, requant);
      }} else {{
        tiu::fmm2_nn(sub_res, sub_left_1, sub_right_1, bias, true, DT_FP32,
                     {do_relu_int} != 0, saturate, requant);
      }}

      tiu::cast(res_{in_type}, sub_res);
      dim4 res_offset = {{0, idx_m, 0, idx_n}};
      dma::store(res_gtensor.sub_view(res_max_shape, res_offset), res_{in_type});
    }}
  }}
}}

__TEST__ void {kernel_name}_main() {{
  const int M = {M};
  const int K = {K};
  const int N = {N};
  dim4 res_shape   = {{1, M, 1, N}};
  dim4 left_shape  = {{1, M, 1, K}};
  dim4 right_shape = {{1, K, 1, N}};
  {in_type} *res   = malloc<{in_type}>(&res_shape);
  rand(res, &res_shape, -1.0, 1.0);
  {in_type} *left  = malloc<{in_type}>(&left_shape);
  rand(left, &left_shape, -1.0, 1.0);
  {in_type} *right = malloc<{in_type}>(&right_shape);
  rand(right, &right_shape, -1.0, 1.0);
  {kernel_name}(res, left, right, M, K, N);
}}
"""


# The ctypes wrapper .cpp.  Built as lib<kernel>_py.so; loaded from Python.
# Mirrors test_tl_gemm_relu/tl_py_wrapper.cpp (hardware-verified).
_WRAPPER_TEMPLATE = r"""// ctypes-callable wrapper around the generated {kernel_name} kernel launch.
#include "{kernel_name}.h"
#include <tpuv7_rt.h>
#include <tpuDNN.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static tpuRtStream_t g_stream = nullptr;
static tpuRtKernelModule_t g_module = nullptr;
// True when g_stream belongs to the caller (py_init_device_shared): the caller
// owns its lifetime, so this wrapper must not destroy it.
static bool g_stream_borrowed = false;

// The kernel module is per-.so, but the stream need not be: every tilelang
// wrapper can share one, which is how torch_tpu itself runs (one stream per
// device, one module per kernel).  Callers that want cross-kernel ordering pass
// the same stream to every wrapper.
static void *init_device_on(int devid, tpuRtStream_t stream) {{
  if (tpuRtInit() != tpuRtSuccess) {{ fprintf(stderr, "tpuRtInit failed\n"); return nullptr; }}
  tpuRtSetDevice(devid);
  if (stream) {{ g_stream = stream; g_stream_borrowed = true; }}
  else {{ g_stream_borrowed = false; tpuRtStreamCreate(&g_stream); }}
  const char *kp = getenv("PPL_KERNEL_PATH");
  if (!kp) {{ fprintf(stderr, "PPL_KERNEL_PATH not set\n"); return nullptr; }}
  g_module = tpuRtKernelLoadModuleFile(kp, g_stream);
  if (!g_module) {{ fprintf(stderr, "tpuRtKernelLoadModuleFile failed for %s\n", kp); return nullptr; }}
  return (void *)tpudnnHandleFromStream(devid, g_stream, g_module);
}}

extern "C" {{

void *py_init_device(int devid) {{ return init_device_on(devid, nullptr); }}
void *py_init_device_shared(int devid, void *stream) {{
  return init_device_on(devid, (tpuRtStream_t)stream);
}}

void py_release_device(void *h) {{
  if (h) tpudnnDestroy((tpudnnHandle_t)h);
  if (g_module) tpuRtKernelUnloadModule(g_module, g_stream);
  if (g_stream && !g_stream_borrowed) {{
    tpuRtStreamSynchronize(g_stream); tpuRtStreamDestroy(g_stream);
  }}
  g_stream = nullptr; g_module = nullptr; g_stream_borrowed = false;
}}

void py_sync_device(void *h) {{ if (h) tpudnnSync((tpudnnHandle_t)h); }}
int py_flush(void *h) {{ return h ? (int)tpudnnFlush((tpudnnHandle_t)h) : -1; }}

unsigned long long py_dev_malloc(unsigned long long size) {{
  void *p = nullptr; tpuRtMalloc(&p, size, 1); return (unsigned long long)p;
}}
void py_dev_free(unsigned long long addr) {{ void *p = (void *)addr; tpuRtFree(&p, 1); }}
void py_memcpy_h2d(unsigned long long dst, const void *src, unsigned long long size) {{ tpuRtMemcpyS2D((void *)dst, src, size); }}
void py_memcpy_d2h(void *dst, unsigned long long src, unsigned long long size) {{ tpuRtMemcpyD2S(dst, (void *)src, size); }}

int py_{kernel_name}(void *h, unsigned long long res, unsigned long long left,
                    unsigned long long right, int M, int K, int N) {{
  return {kernel_name}((tpudnnHandle_t)h, res, left, right, M, K, N);
}}

int py_enable_profile(void *h, int max_record_num, int mode) {{
  return (int)tpudnnEnableProfile((tpudnnHandle_t)h, max_record_num, mode);
}}

int py_disable_profile(void *h) {{
  return (int)tpudnnDisableProfile((tpudnnHandle_t)h);
}}

}}  // extern "C"
"""

# Fragment appended to the --gen_test CMakeLists.txt to build the wrapper .so.
# (Same shape as the tl_py target appended in test_tl_gemm_relu/CMakeLists.txt.)
def _wrapper_cmake_fragment(kernel_name: str, wrapper_src: str) -> str:
    return f"""
# --- tilelang TPU ctypes wrapper (appended by tilelang.tpu.ppl_runner) ---
add_library({kernel_name}_py SHARED {wrapper_src} ${{HOST_SRC_FILES}})
add_dependencies({kernel_name}_py dynamic_library gen_kernel_module_data_target)
target_link_libraries({kernel_name}_py PRIVATE tpudnn ${{RUNTIME_LIBS}} pthread ${{ZLIB_LIBRARIES}} ${{EXTRA_LDFLAGS}})
target_compile_options({kernel_name}_py PRIVATE ${{EXTRA_CFLAGS}})
# Every kernel exports the SAME symbol names (the PPL entry point is always
# ``main_kernel``) and is loaded with RTLD_GLOBAL, so without this a second
# kernel's internal call to ``main_kernel`` binds to the FIRST loaded kernel
# (the global scope is searched before the object's own definitions).  The
# second kernel then runs the first kernel's launch code against its own
# buffers: wrong results, and 100x+ slower.  ``-Bsymbolic`` binds the
# object's own references to its own definitions, making each wrapper
# self-contained.
target_link_options({kernel_name}_py PRIVATE -Wl,-Bsymbolic)
set_target_properties({kernel_name}_py PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS {kernel_name}_py DESTINATION ${{CMAKE_CURRENT_SOURCE_DIR}}/lib)
"""


#: Bump when the wrapper source / CMake layout changes.  It is mixed into the
#: content-hash that names a kernel's build dir (see ``compiler.py``), so a
#: change here rebuilds the cached workdirs instead of silently reusing a .so
#: built with the old layout.
WRAPPER_ABI = "shared-stream-1"


@dataclass
class PPLGemmSpec:
    """Shape/dtype spec for the fp16 GEMM(+ReLU) PPL kernel."""

    M: int
    K: int
    N: int
    block_m: int = 64
    block_k: int = 64
    block_n: int = 64
    relu: bool = True
    in_dtype: str = "fp16"  # "fp16" (verified) or "bf16" (wild-guess, correctness N/A)
    core_num: int = 1       # 1 = single-core, >1 = multi-core (SG2260E has 4)
    num_stages: int = 1     # 1 = single-buffer, 2 = double-buffer (explicit ping-pong)
    kernel_name: str = "tl_gemm_relu"

    def __post_init__(self) -> None:
        if self.in_dtype not in ("fp16", "bf16"):
            raise ValueError(f"in_dtype must be 'fp16' or 'bf16', got {self.in_dtype!r}")
        for name, (val, blk) in {
            "M": (self.M, self.block_m),
            "K": (self.K, self.block_k),
            "N": (self.N, self.block_n),
        }.items():
            if val % blk != 0:
                raise ValueError(
                    f"{name}={val} must be divisible by its tile {blk} "
                    f"(no boundary handling in the simplest path)."
                )


# --------------------------------------------------------------------------- #
# Build                                                                        #
# --------------------------------------------------------------------------- #


def emit_pl(func: Any, out_idx: int | list[int] | None = None) -> "PPLKernelInfo":
    """Translate a lowered tilelang PrimFunc into a PPL kernel (generic path).

    This is the single generic codegen entry point: it walks the lowered TIR
    op-by-op (``tl.tileop.copy/fill/gemm/reduce`` + ``T.Parallel`` elementwise
    bodies) and emits PPL -- no per-family template.  Returns a
    :class:`~tilelang.tpu.ppl_codegen.PPLKernelInfo` carrying the ``.pl`` source
    and the ordered pointer-argument metadata the runtime binds against.

    *out_idx* names the output param(s); it is forwarded from
    ``tilelang.tpu.compiler.compile`` because the ``@T.prim_func`` lowering path
    never materializes the ``tilelang_out_idx`` attr the eager path sets.
    """
    return translate(func, out_idx=out_idx)


def _emit_gemm_template(spec: PPLGemmSpec) -> str:
    """Render the legacy hand-written GEMM(+ReLU) PPL template (int-arg kernel).

    Retained for ``compile_gemm`` / ``run_profiling`` (runtime-int-shape,
    multi-core, multi-buffer variants that the generic translator does not
    model).  The generic ``.compile()`` path uses :func:`emit_pl` instead.
    """
    if spec.num_stages >= 2:
        template = _PL_TEMPLATE_MULTICORE_MULTIBUF if spec.core_num > 1 else _PL_TEMPLATE_MULTIBUF
    else:
        template = _PL_TEMPLATE_MULTICORE if spec.core_num > 1 else _PL_TEMPLATE
    return template.format(
        kernel_name=spec.kernel_name,
        in_type=spec.in_dtype,
        block_m=spec.block_m,
        block_k=spec.block_k,
        block_n=spec.block_n,
        do_relu_int=1 if spec.relu else 0,
        M=spec.M,
        K=spec.K,
        N=spec.N,
    )


def _find_ppl_compile_py(ppl_root: str) -> str | None:
    """Locate the ``ppl_compile.py`` script inside the PPL tree (fallback path)."""
    p = os.path.join(ppl_root, "python", "tool", "ppl_compile.py")
    if os.path.isfile(p):
        return p
    return shutil.which("ppl_compile.py")


def _build_via_ppl_compile_py(
    ppl_root: str,
    pl_path: str,
    workdir: str,
    chip: str,
    opt: str,
    verbose: bool,
) -> None:
    """Fallback: shell out to ``ppl_compile.py`` as a subprocess (original path).

    ``--devid`` is intentionally omitted — it only affects test execution inside
    ``ppl_compile.py``, not code generation, and we never run the test from here.
    """
    script = _find_ppl_compile_py(ppl_root)
    if not script:
        raise RuntimeError(
            "ppl_compile.py not found in the PPL tree and ppl-compile binary "
            "is also missing.  Check your PPL_PROJECT_ROOT."
        )
    cmd = [
        "python3", script,
        "--src", pl_path,
        "--chip", chip,
        "--mode", "pcie",
        "--rv",
        "--opt", opt,
        "--gen_test",
        "--out", workdir,
    ]
    if verbose:
        print("[ppl_runner] fallback ppl_compile.py:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=workdir)


def _generic_wrapper_src(kernel_name: str, num_args: int, num_dyn_dims: int = 0) -> str:
    """Build a ctypes wrapper .cpp for a translator-generated kernel.

    The generic ``__KERNEL__`` takes exactly ``num_args`` pointer arguments (in
    PrimFunc param order; shapes are baked at compile time) followed by
    ``num_dyn_dims`` ``int`` runtime extents for any symbolic buffer dimension.
    The launch entry ``py_<kernel>`` therefore forwards a handle, ``num_args``
    device addresses, and that many dimension values.
    Device-management helpers are identical to the GEMM ``_WRAPPER_TEMPLATE``.
    """
    dim_params = "".join(f", int d{i}" for i in range(num_dyn_dims))
    ptr_params = ", ".join(f"unsigned long long a{i}" for i in range(num_args))
    ptr_args = ", ".join(f"a{i}" for i in range(num_args))
    dim_args = "".join(f", d{i}" for i in range(num_dyn_dims))
    launch = (
        f"int py_{kernel_name}(void *h, {ptr_params}{dim_params}) {{\n"
        f"  return {kernel_name}((tpudnnHandle_t)h, {ptr_args}{dim_args});\n"
        f"}}\n"
    )
    return f"""// ctypes-callable wrapper around the generated {kernel_name} kernel launch.
#include "{kernel_name}.h"
#include <tpuv7_rt.h>
#include <tpuDNN.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static tpuRtStream_t g_stream = nullptr;
static tpuRtKernelModule_t g_module = nullptr;
// True when g_stream belongs to the caller (py_init_device_shared): the caller
// owns its lifetime, so this wrapper must not destroy it.
static bool g_stream_borrowed = false;

// The kernel module is per-.so, but the stream need not be: every tilelang
// wrapper can share one, which is how torch_tpu itself runs (one stream per
// device, one module per kernel).  Callers that want cross-kernel ordering pass
// the same stream to every wrapper.
static void *init_device_on(int devid, tpuRtStream_t stream) {{
  if (tpuRtInit() != tpuRtSuccess) {{ fprintf(stderr, "tpuRtInit failed\\n"); return nullptr; }}
  tpuRtSetDevice(devid);
  if (stream) {{ g_stream = stream; g_stream_borrowed = true; }}
  else {{ g_stream_borrowed = false; tpuRtStreamCreate(&g_stream); }}
  const char *kp = getenv("PPL_KERNEL_PATH");
  if (!kp) {{ fprintf(stderr, "PPL_KERNEL_PATH not set\\n"); return nullptr; }}
  g_module = tpuRtKernelLoadModuleFile(kp, g_stream);
  if (!g_module) {{ fprintf(stderr, "tpuRtKernelLoadModuleFile failed for %s\\n", kp); return nullptr; }}
  return (void *)tpudnnHandleFromStream(devid, g_stream, g_module);
}}

extern "C" {{

void *py_init_device(int devid) {{ return init_device_on(devid, nullptr); }}
void *py_init_device_shared(int devid, void *stream) {{
  return init_device_on(devid, (tpuRtStream_t)stream);
}}

void py_release_device(void *h) {{
  if (h) tpudnnDestroy((tpudnnHandle_t)h);
  if (g_module) tpuRtKernelUnloadModule(g_module, g_stream);
  if (g_stream && !g_stream_borrowed) {{
    tpuRtStreamSynchronize(g_stream); tpuRtStreamDestroy(g_stream);
  }}
  g_stream = nullptr; g_module = nullptr; g_stream_borrowed = false;
}}

void py_sync_device(void *h) {{ if (h) tpudnnSync((tpudnnHandle_t)h); }}
int py_flush(void *h) {{ return h ? (int)tpudnnFlush((tpudnnHandle_t)h) : -1; }}

unsigned long long py_dev_malloc(unsigned long long size) {{
  void *p = nullptr; tpuRtMalloc(&p, size, 1); return (unsigned long long)p;
}}
void py_dev_free(unsigned long long addr) {{ void *p = (void *)addr; tpuRtFree(&p, 1); }}
void py_memcpy_h2d(unsigned long long dst, const void *src, unsigned long long size) {{ tpuRtMemcpyS2D((void *)dst, src, size); }}
void py_memcpy_d2h(void *dst, unsigned long long src, unsigned long long size) {{ tpuRtMemcpyD2S(dst, (void *)src, size); }}

{launch}
int py_enable_profile(void *h, int max_record_num, int mode) {{
  return (int)tpudnnEnableProfile((tpudnnHandle_t)h, max_record_num, mode);
}}

int py_disable_profile(void *h) {{
  return (int)tpudnnDisableProfile((tpudnnHandle_t)h);
}}

}}  // extern "C"
"""


def _build_common(
    pl_content: str,
    kernel_name: str,
    wrapper_text: str,
    workdir: str,
    *,
    chip: str = "sg2260e",
    opt: str = "O3",
    verbose: bool = False,
) -> dict[str, str]:
    """Emit .pl + ctypes wrapper, compile device code, cmake-build the .so's.

    Content-hash cached on ``pl_content``.  Returns
    ``{"pl", "kernel_so", "wrapper_so", "test_case", "workdir"}``.
    """
    logger.warning("[TPU]: ppl_runner->build()")
    ppl_root = _get_ppl_root()
    chip_arch = _resolve_chip_arch(ppl_root, chip)

    os.makedirs(workdir, exist_ok=True)
    pl_path = os.path.join(workdir, f"{kernel_name}.pl")
    kernel_so = os.path.join(workdir, "lib", "libkernel.so")
    wrapper_so = os.path.join(workdir, "lib", f"{kernel_name}_py.so")
    test_case = os.path.join(workdir, "test_case")

    cached = False
    if (os.path.isfile(kernel_so) and os.path.isfile(wrapper_so)
            and os.path.isfile(pl_path)):
        with open(pl_path) as f:
            if f.read() == pl_content:
                cached = True
                logger.warning("[TPU]: ppl_runner->build() cache HIT, skipping compilation")

    if cached:
        tc = test_case if os.path.isfile(test_case) else None
        return {"pl": pl_path, "kernel_so": kernel_so, "wrapper_so": wrapper_so,
                "test_case": tc, "workdir": workdir}

    with open(pl_path, "w") as f:
        f.write(pl_content)
    logger.warning(f"[TPU]: ppl_runner->build(), emit_pl() into {pl_path}")

    compiler_bin = os.path.join(ppl_root, "bin", "ppl-compile")
    if os.path.isfile(compiler_bin):
        # The build sets CHIP/CHIP_ARCH to the chip CODE; scope that so it does
        # not outlive the build and poison the runtime's arch-name lookup.
        with _scoped_chip_env(("CHIP", "CHIP_ARCH")):
            _setup_ppl_env(ppl_root, chip, chip_arch, workdir, kernel_name)
            _run_ppl_compile(ppl_root, pl_path, chip_arch, workdir, opt=opt, rv=True,
                             autotune=True, verbose=verbose)
            _cmake_build(ppl_root, chip_arch, workdir, mode="pcie", verbose=verbose)
    else:
        logger.warning("[TPU]: ppl-compile binary not found, falling back to ppl_compile.py subprocess")
        with _scoped_chip_env(("CHIP", "CHIP_ARCH")):
            _build_via_ppl_compile_py(ppl_root, pl_path, workdir, chip, opt, verbose)

    if not os.path.isfile(kernel_so):
        raise RuntimeError(f"libkernel.so not produced at {kernel_so}")

    wrapper_src = f"{kernel_name}_py_wrapper.cpp"
    wrapper_path = os.path.join(workdir, wrapper_src)
    with open(wrapper_path, "w") as f:
        f.write(wrapper_text)

    cmake_path = os.path.join(workdir, "CMakeLists.txt")
    with open(cmake_path, "a") as f:
        f.write(_wrapper_cmake_fragment(kernel_name, wrapper_src))

    build_dir = os.path.join(workdir, "build")
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        ["cmake", "-S", workdir, "-B", build_dir],
        check=True, cwd=workdir,
    )
    subprocess.run(
        ["cmake", "--build", build_dir, "--target", f"{kernel_name}_py", "-j"],
        check=True, cwd=workdir,
    )
    subprocess.run(
        ["cmake", "--build", build_dir, "--target", "install"],
        check=True, cwd=workdir,
    )
    if not os.path.isfile(wrapper_so):
        raise RuntimeError(f"wrapper .so not found at {wrapper_so}")

    tc = test_case if os.path.isfile(test_case) else None
    logger.warning(f"[TPU]: ppl_runner->build() done, kernel_so = {kernel_so}, wrapper_so = {wrapper_so}")
    return {"pl": pl_path, "kernel_so": kernel_so, "wrapper_so": wrapper_so,
            "test_case": tc, "workdir": workdir}


def build(
    spec: "PPLGemmSpec",
    workdir: str,
    *,
    chip: str = "sg2260e",
    opt: str = "O3",
    verbose: bool = False,
) -> dict[str, str]:
    """Build the legacy hand-written GEMM template kernel (``compile_gemm`` path)."""
    wrapper_text = _WRAPPER_TEMPLATE.format(kernel_name=spec.kernel_name)
    return _build_common(
        _emit_gemm_template(spec), spec.kernel_name, wrapper_text, workdir,
        chip=chip, opt=opt, verbose=verbose,
    )


def build_generic(
    info: "PPLKernelInfo",
    workdir: str,
    *,
    chip: str = "sg2260e",
    opt: str = "O3",
    verbose: bool = False,
) -> dict[str, str]:
    """Build a translator-generated kernel (generic ``compile`` path)."""
    wrapper_text = _generic_wrapper_src(info.kernel_name, len(info.args), len(info.dyn_dims))
    return _build_common(
        info.source, info.kernel_name, wrapper_text, workdir,
        chip=chip, opt=opt, verbose=verbose,
    )


# --------------------------------------------------------------------------- #
# Profiling                                                                    #
# --------------------------------------------------------------------------- #


def _parse_summary(summary_path: str) -> float:
    """Parse bigTpuProfile summary.txt for the Overall time (us)."""
    if not os.path.isfile(summary_path):
        return 0.0
    import ast as _ast
    with open(summary_path) as f:
        for line in f:
            if "Overall" in line:
                parts = line.split("|")
                if len(parts) >= 3:
                    return float(_ast.literal_eval(parts[2]))
    return 0.0


def _select_cdm_dump(
    profiling_dir: str, device: int | None, *,
    verbose: bool = False,
) -> list[str]:
    """Pick the dump(s) in *profiling_dir* that belong to *device*.

    The runtime names each dump after the device that produced it
    (``cdm_profile_data_dev7-0``) and drops it in the cwd, which
    ``_move_profile_data`` then sweeps into the profile dir.  Because that move
    only replaces a dump of the *same* name, running a second profile session on
    a different device leaves both behind -- and processing both yields two
    summaries for what the user sees as one ``--profile`` run.  Selecting by
    device id keeps that from happening.

    Falls back to every dump found when *device* is ``None`` or matches nothing,
    so an unexpected naming scheme still yields (possibly over-complete) data
    rather than a hard failure.
    """
    import glob as _glob
    import sys as _sys

    all_dumps = sorted(_glob.glob(os.path.join(profiling_dir, "cdm_profile_data_dev*")))
    if device is None or len(all_dumps) <= 1:
        return all_dumps

    want = os.path.join(profiling_dir, f"cdm_profile_data_dev{device}-*")
    mine = sorted(_glob.glob(want))
    if not mine:
        if verbose:
            print(
                f"[ppl_runner] no dump for device {device}; "
                f"falling back to all {len(all_dumps)} dump(s)",
                file=_sys.stderr,
            )
        return all_dumps
    return mine


def _collect_profile_data(
    profiling_dir: str, *, device: int | None = None, verbose: bool = False,
) -> dict[str, Any]:
    """Process ``cdm_profile_data_dev*`` files with ``bigTpuProfile``.

    Only dumps produced by *device* are processed (see
    :func:`_select_cdm_dump`); pass ``device=None`` to process all of them.

    If ``bigTpuProfile`` fails (e.g. timestamp normalization errors), the raw
    ``cdm_profile_data_dev*`` files are still returned so the user can inspect
    or reprocess them manually.  A warning is printed instead of raising.
    """
    import sys as _sys
    import warnings as _warnings

    cdm_files = _select_cdm_dump(profiling_dir, device, verbose=verbose)
    if not cdm_files:
        raise RuntimeError(f"No cdm_profile_data_dev* files found in {profiling_dir}")

    overall_us = 0.0
    summary_path = ""
    pftrace_path = ""
    parse_ok = True

    for i, cdm_file in enumerate(cdm_files):
        out_dir = os.path.join(profiling_dir, f"out_{i}")
        cmd = ["bigTpuProfile", cdm_file, out_dir]
        if verbose:
            print(f"[ppl_runner] {' '.join(cmd)}")
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ret.returncode != 0:
            parse_ok = False
            _warnings.warn(
                f"bigTpuProfile failed for {cdm_file} (exit {ret.returncode}). "
                f"Raw data preserved at {cdm_file} for manual inspection.\n"
                f"stdout: {ret.stdout.strip()}\nstderr: {ret.stderr.strip()}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        sp = os.path.join(out_dir, "summary.txt")
        overall_us += _parse_summary(sp)
        if i == 0:
            summary_path = sp
            pf = os.path.join(out_dir, "perfetto.pftrace")
            if os.path.isfile(pf):
                pftrace_path = pf
        if os.path.isfile(sp):
            print(open(sp).read(), file=_sys.stderr)

    print("=" * 60, file=_sys.stderr)
    if parse_ok:
        print(f"  Overall kernel time: {overall_us:.2f} us", file=_sys.stderr)
    else:
        print("  WARNING: bigTpuProfile failed to parse some/all data", file=_sys.stderr)
        print("  Raw cdm files are preserved for manual inspection.", file=_sys.stderr)
    print(f"  Profiling data:      {profiling_dir}", file=_sys.stderr)
    print(f"  Raw cdm files:       {', '.join(cdm_files)}", file=_sys.stderr)
    if pftrace_path:
        print(f"  Perfetto trace:      {pftrace_path}", file=_sys.stderr)
        print("  Open in Perfetto UI: https://ui.perfetto.dev/", file=_sys.stderr)
    print("=" * 60, file=_sys.stderr)

    return {
        "profiling_dir": profiling_dir,
        "overall_us": overall_us,
        "summary_path": summary_path,
        "pftrace_path": pftrace_path,
        "cdm_files": cdm_files,
        "parse_ok": parse_ok,
    }


def run_profiling(
    workdir: str,
    *,
    spec: PPLGemmSpec | None = None,
    devid: int | None = None,
    book_keeping: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run kernel with hardware profiling via the ctypes execution path.

    Uses the same ``PPLKernel`` ctypes wrapper as normal kernel execution —
    no separate ``test_case`` binary needed.  ``tpudnnEnableProfile()`` is
    called on the device handle before the kernel launch so that a single
    run produces both a result tensor and profiling data.

    When *spec* is given, its M/K/N are used for the profiling run (important
    for dynamic-shape kernels).  Otherwise defaults to 1024x1024x1024.

    Returns ``{"profiling_dir", "overall_us", "summary_path", "pftrace_path"}``.
    """
    if devid is None:
        devid = get_tpu_device()

    M = spec.M if spec else 1024
    K = spec.K if spec else 1024
    N = spec.N if spec else 1024
    in_dtype = spec.in_dtype if spec else "fp16"
    kernel_name = spec.kernel_name if spec else "tl_gemm_relu"

    kernel_so = os.path.join(workdir, "lib", "libkernel.so")
    wrapper_so = os.path.join(workdir, "lib", f"{kernel_name}_py.so")
    if not os.path.isfile(kernel_so) or not os.path.isfile(wrapper_so):
        raise RuntimeError(
            f"Kernel not built in {workdir} — run build() first. "
            f"Missing: {kernel_so if not os.path.isfile(kernel_so) else wrapper_so}"
        )

    paths = {"kernel_so": kernel_so, "wrapper_so": wrapper_so, "workdir": workdir}
    profiling_dir = os.path.join(workdir, "profiling")
    os.makedirs(profiling_dir, exist_ok=True)

    os.environ["BMLIB_ENABLE_ALL_PROFILE"] = "1"
    os.environ["PROFILE_BOOK_KEEPING"] = str(book_keeping)

    _ensure_tpu_env(kernel_name)
    os.environ["PPL_KERNEL_PATH"] = kernel_so

    kernel = PPLKernel(paths, device=devid, kernel_name=kernel_name)
    kernel.init()
    kernel.enable_profile(book_keeping=book_keeping)
    kernel._profile_dir = profiling_dir

    dtype = torch.float16 if in_dtype == "fp16" else torch.bfloat16
    a = torch.randn(M, K, dtype=dtype)
    b = torch.randn(K, N, dtype=dtype)

    logger.warning(f"[TPU]: profiling {kernel_name} M={M} K={K} N={N} in {profiling_dir}")
    kernel.run(a, b)
    kernel.close()

    return _collect_profile_data(profiling_dir, device=devid, verbose=verbose)


# --------------------------------------------------------------------------- #
# Runtime                                                                      #
# --------------------------------------------------------------------------- #


# Real-TPU (pcie) runtime env.  Discovered by gdb'ing the exit-134
# "basic_string: construction from null" failure: the driver reads these via
# getenv and turns them into std::string; the npz_save path also needs
# PPL_FILE_NAME (release builds compile out the assert, so NULL -> throw).
_TPUV7_CURRENT = "/opt/tpuv7/tpuv7-current"
_TPUV7_LIB = _TPUV7_CURRENT + "/lib"


def _ensure_tpu_env(kernel_name: str) -> None:
    """Populate the env vars the real SG2260E driver + npz_save require."""
    def _set(k: str, v: str) -> None:
        if not os.environ.get(k):
            os.environ[k] = v

    _set("TPU_OPER_PATH", _TPUV7_CURRENT + "/data")
    _set("AKS_MODULE_PATH", _TPUV7_CURRENT + "/data/AKS/libfirmware_core.so")
    _set("AKSV_MODULE_PATH", _TPUV7_CURRENT + "/data/AKSV/libfirmware_core.so")
    _set("PPL_TPUKERNEL_DEV_MODE", "pcie")
    _set("PPL_FILE_NAME", kernel_name)

    ppl_root = _get_ppl_root()
    chip_lib = os.path.join(ppl_root, "deps", "chip", "tpub_7_1_e", "lib")
    rt_lib = os.path.join(ppl_root, "deps", "runtime", "tpuv7-runtime", "lib")
    ld_existing = os.environ.get("LD_LIBRARY_PATH", "")
    present = set(ld_existing.split(os.pathsep)) if ld_existing else set()
    to_prepend = [d for d in (_TPUV7_LIB, chip_lib, rt_lib) if d and d not in present]
    if to_prepend:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            to_prepend + ([ld_existing] if ld_existing else [])
        )


def _preload_real_driver() -> None:
    """Preload shared libs (RTLD_GLOBAL) before the wrapper .so.

    dlopen only reads LD_LIBRARY_PATH at process startup, so runtime changes
    via os.environ are too late.  We explicitly preload every .so the wrapper
    links against:

    * libtpuv7_rt.so / libtpuv7_modelrt.so from /opt/tpuv7/.../lib — the REAL
      driver, not the emulator copy under deps/runtime/.
    * libtpudnn.so from the PPL chip lib dir — TPU DNN ops used by the wrapper.
    """
    for name in ("libtpuv7_rt.so", "libtpuv7_modelrt.so"):
        path = os.path.join(_TPUV7_LIB, name)
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass

    try:
        ppl_root = _get_ppl_root()
        chip_lib = os.path.join(ppl_root, "deps", "chip", "tpub_7_1_e", "lib")
        for name in ("libtpudnn.so",):
            path = os.path.join(chip_lib, name)
            if os.path.isfile(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
    except RuntimeError:
        pass


class PPLKernel:
    """ctypes handle to a built PPL GEMM(+ReLU) kernel on TPU."""

    def __init__(self, paths: dict[str, str], *, device: int = 0, kernel_name: str = "tl_gemm_relu"):
        logger.warning("[TPU]: PPLKernel->__init__")
        self.paths = paths
        self.device = device
        self.kernel_name = kernel_name
        _ensure_tpu_env(kernel_name)
        os.environ["PPL_KERNEL_PATH"] = paths["kernel_so"]
        _preload_real_driver()
        self.lib = ctypes.CDLL(paths["wrapper_so"], mode=ctypes.RTLD_GLOBAL)
        L = self.lib
        kn = kernel_name
        L.py_init_device.argtypes = [ctypes.c_int]
        L.py_init_device.restype = ctypes.c_void_p
        L.py_release_device.argtypes = [ctypes.c_void_p]
        L.py_sync_device.argtypes = [ctypes.c_void_p]
        # Present in wrappers built from this template onwards; a stale build
        # dir simply falls back to the drain+sync launch.
        self._flush = getattr(L, "py_flush", None)
        self._init_device_shared = getattr(L, "py_init_device_shared", None)
        if self._init_device_shared is not None:
            self._init_device_shared.argtypes = [ctypes.c_int, ctypes.c_void_p]
            self._init_device_shared.restype = ctypes.c_void_p
        # torch view of the shared PPL stream; re-wrapped if the pointer changes.
        self._ext_stream = None
        self._ext_stream_ptr = None
        L.py_dev_malloc.argtypes = [ctypes.c_uint64]
        L.py_dev_malloc.restype = ctypes.c_uint64
        L.py_dev_free.argtypes = [ctypes.c_uint64]
        L.py_memcpy_h2d.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        L.py_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        launch = getattr(L, f"py_{kn}")
        launch.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        launch.restype = ctypes.c_int
        self._launch = launch
        L.py_enable_profile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.py_enable_profile.restype = ctypes.c_int
        L.py_disable_profile.argtypes = [ctypes.c_void_p]
        L.py_disable_profile.restype = ctypes.c_int
        self.handle: Any = None
        self._profile_dir: str | None = None
        self._profile_active: bool = False

    def init(self) -> None:
        stream = self._shared_stream()
        if stream is None:
            self.handle = self.lib.py_init_device(self.device)
        else:
            self.handle = self.lib.py_init_device_shared(self.device, stream)
        if not self.handle:
            raise RuntimeError("py_init_device failed (is the TPU device free?)")

    def _shared_stream(self) -> int | None:
        """The device-wide PPL stream, or ``None`` to use a private one."""
        if not _ASYNC_LAUNCH or self._init_device_shared is None:
            return None
        shared = _ppl_shared_stream(self.device)
        return None if shared is None else shared[1]

    def close(self) -> None:
        if self.handle:
            if self._profile_active:
                # py_disable_profile launches an sg_api_set_profile kernel; a
                # torch_tpu op since the last launch desyncs the runtime device,
                # so re-assert before the launch (see _reassert_tpu_device).
                _reassert_tpu_device(self.device)
                self.lib.py_disable_profile(self.handle)
                self._profile_active = False
                self._move_profile_data()
            self.lib.py_release_device(self.handle)
            self.handle = None
            # Drop the torch view: a re-init may bind a different stream.
            self._ext_stream = None
            self._ext_stream_ptr = None

    def _move_profile_data(self) -> None:
        """Move ``cdm_profile_data_dev*`` from cwd to ``_profile_dir``."""
        if not self._profile_dir:
            return
        import glob as _glob
        for src in _glob.glob("cdm_profile_data_dev*"):
            dst = os.path.join(self._profile_dir, os.path.basename(src))
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            shutil.move(src, dst)

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *a):
        self.close()

    def enable_profile(self, max_record_num: int = 0, book_keeping: int = 1) -> None:
        """Call ``tpudnnEnableProfile`` on the device handle.

        ``BMLIB_ENABLE_ALL_PROFILE=1`` must be set in the environment *before*
        ``init()`` (i.e. before ``tpuRtInit``).  This method calls the runtime
        API to start recording profiling data for subsequent kernel launches.
        """
        if not self.handle:
            raise RuntimeError("Call init() before enable_profile()")
        if max_record_num <= 0:
            env_val = os.environ.get("PROFILE_RECORD_SIZE", "")
            max_record_num = int(env_val) if env_val else 131072
        # py_enable_profile launches an sg_api_set_profile kernel; re-assert the
        # runtime device first (a torch_tpu op may have desynced it).
        _reassert_tpu_device(self.device)
        ret = self.lib.py_enable_profile(self.handle, max_record_num, book_keeping)
        if ret != 0:
            raise RuntimeError(f"tpudnnEnableProfile failed with code {ret}")
        self._profile_active = True

    def collect_profile(self, *, verbose: bool = False) -> dict[str, Any]:
        """Process ``cdm_profile_data_dev*`` files with ``bigTpuProfile``."""
        if not self._profile_dir:
            raise RuntimeError("Profiling not enabled — no profile_dir set")
        return _collect_profile_data(
            self._profile_dir, device=self.device, verbose=verbose
        )

    def run(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """{fp16|bf16} [M,K] x [K,N] -> [M,N] (with ReLU if compiled with relu).

        Device-pointer launch: ``a`` and ``b`` must be TPU device tensors.  The
        output is allocated on the device and written in place by the kernel;
        nothing crosses host memory.
        """
        logger.warning("[TPU]: PPLKernel->run()")
        assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
        c_addr = int(c.data_ptr())
        a_addr = int(a.data_ptr())
        b_addr = int(b.data_ptr())
        _reassert_tpu_device(self.device)
        _sync_torch_stream()
        if self._launch(self.handle, c_addr, a_addr, b_addr, M, K, N) != 0:
            raise RuntimeError("kernel launch failed")
        self.lib.py_sync_device(self.handle)
        return c


def _concrete_dim(s: Any, dims_by_name: dict[str, int]) -> int:
    """Resolve one output-shape entry: a symbolic var's value, or the literal."""
    name = str(s)
    if name in dims_by_name:
        return dims_by_name[name]
    try:
        return int(s)
    except (TypeError, ValueError):
        raise ValueError(
            f"output dim {name!r} could not be resolved to a concrete value")


class PPLGenericKernel(PPLKernel):
    """ctypes handle to a translator-generated PPL kernel on TPU.

    Reuses :class:`PPLKernel`'s device management (init/close/profile),
    but binds the launch signature and ``run`` generically from a
    :class:`~tilelang.tpu.ppl_codegen.PPLKernelInfo`: N pointer arguments in
    PrimFunc param order (inputs + pre-allocated outputs), followed by one
    ``int`` per symbolic buffer dimension.  The dim values are resolved from the
    input tensors' actual shapes at each call, so a single build serves any
    shape the tiles divide evenly.
    """

    def __init__(self, paths: dict[str, str], *, device: int = 0,
                 kernel_name: str = "tl_kernel", info: PPLKernelInfo | None = None):
        super().__init__(paths, device=device, kernel_name=kernel_name)
        assert info is not None, "PPLGenericKernel requires a PPLKernelInfo"
        self.info = info
        # Inputs of the current run, kept only so _resolve_dims can read their
        # shapes; cleared once the launch returns.
        self._inputs: list[torch.Tensor] = []
        # Base __init__ bound the GEMM launch signature; rebind for N pointers
        # plus the runtime dims.
        launch = getattr(self.lib, f"py_{kernel_name}")
        launch.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.c_uint64] * len(info.args)
            + [ctypes.c_int] * len(info.dyn_dims)
        )
        launch.restype = ctypes.c_int
        self._launch = launch

    def _resolve_dims(self) -> dict[str, int]:
        """Map each symbolic dim name to a value from the input shapes.

        Called after the inputs have been materialized; ``self._inputs`` holds
        them in ``info.inputs`` order.  Walks ``info.dyn_slots`` -- the
        ``(extent position, dim index)`` pairs derived once per kernel -- so no
        extent is stringified per launch (see that property).
        """
        info = self.info
        slots_by_input = [s for a, s in zip(info.args, info.dyn_slots)
                          if not a.is_output]
        found: list[int | None] = [None] * len(info.dyn_dims)
        for slots, t in zip(slots_by_input, self._inputs):
            logical = tuple(t.shape)
            for pos, k in slots:
                name = info.dyn_dims[k]
                if pos >= len(logical):
                    raise ValueError(
                        f"cannot resolve dim {name!r} from a tensor of shape "
                        f"{logical}")
                val = int(logical[pos])
                # A symbolic name may recur across buffers (e.g. B in both Q and
                # K); every occurrence must agree, else the caller passed
                # inconsistent shapes and first-buffer-wins would silently pick
                # one. Fail loudly instead.
                if found[k] is not None and found[k] != val:
                    raise ValueError(
                        f"dynamic dim {name!r} has conflicting values "
                        f"{found[k]} and {val} across input buffers for "
                        f"kernel {self.kernel_name}; check the input shapes")
                found[k] = val
        missing = [n for n, v in zip(info.dyn_dims, found) if v is None]
        if missing:
            raise ValueError(
                f"unresolved dynamic dim(s) {missing} for kernel "
                f"{self.kernel_name}; pass tensors whose shapes cover them")
        return dict(zip(info.dyn_dims, found))

    def run(self, *inputs: torch.Tensor):
        """Launch with device tensors; allocate outputs on-device, no H2D/D2H.

        ``inputs`` are TPU device tensors in the order of ``info.inputs`` (the
        non-output params).  Returns the single output tensor, or a tuple when
        the kernel declares multiple outputs — always device tensors.

        This is a thin wrapper over :meth:`run_device_ptrs`: it resolves the
        dynamic dims from the input shapes, allocates any output the caller did
        not supply with torch_tpu's allocator, and hands raw device addresses to
        the kernel.  Nothing is copied through host memory.
        """
        in_args = self.info.inputs
        if not (len(in_args) <= len(inputs) <= len(in_args) + len(self.info.outputs)):
            raise ValueError(
                f"{self.kernel_name} expects {len(in_args)} inputs (plus up to "
                f"{len(self.info.outputs)} preallocated outputs), got {len(inputs)}"
            )
        device_inputs = list(inputs[: len(in_args)])
        provided = list(inputs[len(in_args) :])

        for t in device_inputs:
            if t.device.type != "tpu":
                raise ValueError(
                    f"{self.kernel_name}: inputs must be TPU device tensors "
                    f"(the host-marshalling path was removed); got "
                    f"'{t.device.type}'"
                )

        self._inputs = device_inputs
        try:
            dims_by_name = self._resolve_dims()
            tpu_device = f"tpu:{get_torch_device()}"
            out_tensors: list[torch.Tensor] = []
            for i, arg in enumerate(self.info.outputs):
                if i < len(provided):
                    out_tensors.append(provided[i])
                else:
                    shape = tuple(_concrete_dim(s, dims_by_name) for s in arg.shape)
                    out_tensors.append(
                        torch.empty(
                            shape,
                            dtype=_TORCH_DTYPE[arg.torch_dtype],
                            device=tpu_device,
                        )
                    )
            self.run_device_ptrs(
                [int(t.data_ptr()) for t in device_inputs],
                dims_by_name,
                [int(t.data_ptr()) for t in out_tensors],
            )
        finally:
            self._inputs = []

        return out_tensors[0] if len(out_tensors) == 1 else tuple(out_tensors)

    def _external_stream(self):
        """torch view of the wrapper's PPL stream, or ``None`` if unavailable.

        Returns ``None`` when the async path is off (``PPL_ASYNC_LAUNCH=1``),
        the wrapper predates ``py_flush``/``py_init_device_shared``, or the
        shared stream could not be created -- all of which mean "use the
        drain+sync launch".
        """
        if not _ASYNC_LAUNCH or self._flush is None or self._init_device_shared is None:
            return None
        shared = _ppl_shared_stream(self.device)
        if shared is None:
            return None
        stream, ptr = shared
        self._ext_stream = stream
        self._ext_stream_ptr = ptr
        return self._ext_stream

    def _ordered_launch(self, *args) -> int:
        """Launch so the kernel is ordered against the caller's torch stream.

        torch_tpu enqueues its ops with ``tpuRtLaunchNetAsync`` on the current
        stream and only fences at ``tpuSynchronizeDevice``; the PPL wrapper
        launches on a stream of its own.  Two things follow:

        * The caller's writes to the input buffers are *enqueued*, not done, so
          the PPL stream must wait for the caller's stream (`_sync_torch_stream`
          waits for all of it -- a full drain, 1 ms short-queue and ~4.5 ms with
          a model layer queued).  Waiting on an *event* rather than draining is
          enough, but only after ``torch.tpu.flush``: an event recorded on a
          stream is ordered against what was submitted to the driver, and
          ``tpuRtLaunchNetAsync`` stages everything until a flush.  Recording
          the event on unflushed work lets the kernel read buffers whose writes
          are still sitting in torch's stage queue -- the fused qkv kernel then
          reads a stale ``kv_row`` and writes K/V to the *previous* call's
          slots, which turns the model to garbage (seen in-model; a standalone
          probe hides it because there the inputs arrive by H2D copies).
        * A PPL launch is not visible to the runtime until ``tpudnnFlush``, which
          is exactly what ``tpuSynchronizeDevice`` does before syncing.  Flushing
          right after the launch publishes the kernel on the wrapper's stream,
          which lets one event each way replace the drain and the fence.

        The event pair is what the driver's own ops use, so it is not a weaker
        guarantee: it is the same ordering torch gets, just without a full
        device drain -- provided both sides are published, hence the
        ``torch.tpu.flush`` below and the ``py_flush`` after the launch.

        **This needs the shared stream.**  A first attempt that left each
        wrapper on its own stream (two PPL streams: the fused qkv kernel and the
        paged-attention kernel) hung a model run 90 s in and left the chip
        ``Fault`` forever.  Launches staged on two PPL streams are not ordered
        against each other, and flushing them out of order is the suspected
        cause.  With :func:`_ppl_shared_stream` every kernel on the device
        launches into one stream, so one flush publishes all of them in order --
        the same shape torch_tpu has (one stream per device, many modules).

        **The shared stream must be torch-owned.**  A stream created with
        ``tpuRtStreamCreate`` and wrapped in ``ExternalStream`` still raced: with
        the flush above, four in-model trials gave three plausible continuations
        and one 1.5-nat shifted first two tokens, and a full
        ``torch.tpu.synchronize()`` *before* the launch did not fix it, so it was
        not input staleness.  Creating the stream with ``torch.tpu.Stream`` and
        giving its raw pointer to ``py_init_device_shared`` closed it -- ten
        consecutive trials then matched the torch chain within its own
        run-to-run noise.  A torch-owned stream is a normal stream to the
        runtime, so its events are ordinary stream events rather than
        ``ExternalStream`` fakes.
        """
        ext = self._external_stream()
        if ext is None:
            _sync_torch_stream()
            rc = self._launch(*args)
            self.lib.py_sync_device(self.handle)
            return rc
        cur = torch.tpu.current_stream()
        # Publish the caller's staged torch work before recording the event;
        # see the docstring.  Cheap: it submits, it does not wait.
        torch.tpu.flush()
        ext.wait_stream(cur)
        rc = self._launch(*args)
        if self._flush(self.handle) != 0:
            raise RuntimeError(f"{self.kernel_name}: tpudnnFlush failed")
        cur.wait_stream(ext)
        return rc

    def run_device_ptrs(
        self,
        input_addrs: list[int],
        dims_by_name: dict[str, int],
        output_addrs: list[int],
    ) -> None:
        """Launch the kernel using pre-supplied device addresses, no H2D/D2H.

        ``input_addrs`` and ``output_addrs`` are raw device pointers in the
        order of ``info.inputs`` and ``info.outputs`` respectively.  The caller
        owns all allocation.  On return the kernel is ordered against the
        caller's torch stream (see :meth:`_ordered_launch`): a torch op that
        reads an output waits for the kernel, and vice versa, so no caller-side
        sync is needed.

        ``dims_by_name`` maps every symbolic dim name (as declared with
        ``T.dynamic``) to its concrete integer value for this invocation; see
        :meth:`_resolve_dims` for how the host-tensor path derives the same map.

        This is the device-pointer fast path: tensors that already live on the
        TPU can be passed directly without any host memcpy, which is critical
        for kernels that process large paged KV caches.
        """
        if len(input_addrs) != len(self.info.inputs):
            raise ValueError(
                f"{self.kernel_name} expects {len(self.info.inputs)} input "
                f"addresses, got {len(input_addrs)}"
            )
        if len(output_addrs) != len(self.info.outputs):
            raise ValueError(
                f"{self.kernel_name} expects {len(self.info.outputs)} output "
                f"addresses, got {len(output_addrs)}"
            )
        missing = [n for n in self.info.dyn_dims if n not in dims_by_name]
        if missing:
            raise ValueError(
                f"unresolved dynamic dim(s) {missing} for kernel "
                f"{self.kernel_name}; pass them in dims_by_name"
            )
        # Build the ordered address list: args = inputs + outputs in PrimFunc order.
        addr_by_name: dict[str, int] = {}
        for arg, addr in zip(self.info.inputs, input_addrs):
            addr_by_name[arg.name] = addr
        for arg, addr in zip(self.info.outputs, output_addrs):
            addr_by_name[arg.name] = addr
        addrs = [addr_by_name[a.name] for a in self.info.args]
        dims = [dims_by_name[n] for n in self.info.dyn_dims]
        # The caller allocated the output via torch_tpu after its own device
        # sync, which desyncs the runtime; re-assert immediately before launch.
        _reassert_tpu_device(self.device)
        if self._ordered_launch(self.handle, *addrs, *dims) != 0:
            raise RuntimeError(f"{self.kernel_name} device-ptr launch failed")

