"""PPL build & run engine for tilelang's TPU backend.

This module packages the *verified* recipe for running a tilelang-style GEMM
kernel on SG2260E TPU hardware via the PPL toolchain.  The recipe mirrors the
canonical reference kernel ``PPLOPs/pplops/ops/gemm_w8a8bf16.pl``:

  1. emit a PPL ``.pl`` kernel (``__KERNEL__`` + ``__TEST__``);
  2. ``ppl_compile.py --gen_test`` builds ``lib/libkernel.so`` + a host launcher
     (``host/*.cpp``, ``include/*.h``) + a CMakeLists.txt;
  3. append a small ctypes wrapper target (``tl_py.so``) that exposes
     ``py_init_device / py_sync_device / py_dev_malloc / py_memcpy_h2d /
     py_memcpy_d2h / py_<kernel>`` and rebuild that target;
  4. from Python, ``ctypes``-load ``tl_py.so``, copy torch tensors H2D, launch,
     sync, copy D2H.

The fp16 GEMM+ReLU kernel emitted here (M=N=K=1024, 64x64x64 tiles, fp32 accum)
was compiled and run on real TPU ``devid 2`` and matched ``torch.relu(a @ b)``
within ``rtol=atol=1e-2`` (max diff 0.015625) -- i.e. it passes the assertion in
``test_tilelang/test_gemm_naive.py``.

Only the fp16 -> fp32-accum -> (relu) -> fp16 path is verified end-to-end today.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import torch
import logging

logger = logging.getLogger(__name__)

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

extern "C" {{

void *py_init_device(int devid) {{
  if (tpuRtInit() != tpuRtSuccess) {{ fprintf(stderr, "tpuRtInit failed\n"); return nullptr; }}
  tpuRtSetDevice(devid);
  tpuRtStreamCreate(&g_stream);
  const char *kp = getenv("PPL_KERNEL_PATH");
  if (!kp) {{ fprintf(stderr, "PPL_KERNEL_PATH not set\n"); return nullptr; }}
  g_module = tpuRtKernelLoadModuleFile(kp, g_stream);
  if (!g_module) {{ fprintf(stderr, "tpuRtKernelLoadModuleFile failed for %s\n", kp); return nullptr; }}
  return (void *)tpudnnHandleFromStream(devid, g_stream, g_module);
}}

void py_release_device(void *h) {{
  if (h) tpudnnDestroy((tpudnnHandle_t)h);
  if (g_module) tpuRtKernelUnloadModule(g_module, g_stream);
  if (g_stream) {{ tpuRtStreamSynchronize(g_stream); tpuRtStreamDestroy(g_stream); g_stream = nullptr; }}
}}

void py_sync_device(void *h) {{ if (h) tpudnnSync((tpudnnHandle_t)h); }}

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
set_target_properties({kernel_name}_py PROPERTIES PREFIX "" SUFFIX ".so")
install(TARGETS {kernel_name}_py DESTINATION ${{CMAKE_CURRENT_SOURCE_DIR}}/lib)
"""


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


def _ppl_compile_py() -> str:
    root = os.environ.get("PPL_PROJECT_ROOT")
    if not root:
        raise RuntimeError(
            "PPL_PROJECT_ROOT is not set. Source the PPL envsetup.sh first "
            "(e.g. `source <ppl>/envsetup.sh`)."
        )
    p = os.path.join(root, "python", "tool", "ppl_compile.py")
    if not os.path.isfile(p):
        p = shutil.which("ppl_compile.py") or ""
    if not p:
        raise RuntimeError("ppl_compile.py not found.")
    return p


def emit_pl(spec: PPLGemmSpec) -> str:
    return _PL_TEMPLATE.format(
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


def build(
    spec: PPLGemmSpec,
    workdir: str,
    *,
    chip: str = "sg2260e",
    devid: int = 3,
    opt: str = "O3",
    verbose: bool = False,
) -> dict[str, str]:
    """Emit .pl, run ppl_compile.py --gen_test, then build the ctypes wrapper .so.

    Returns a dict with paths: {"pl", "kernel_so", "wrapper_so", "workdir"}.
    """
    logger.warning("[TPU]: ppl_runner->build()")
    os.makedirs(workdir, exist_ok=True)
    pl_path = os.path.join(workdir, f"{spec.kernel_name}.pl")
    with open(pl_path, "w") as f:
        f.write(emit_pl(spec))
    logger.warning(f"[TPU]: ppl_runner->build(), emit_pl() into {pl_path}")

    # 1) ppl_compile.py --gen_test  ->  workdir/{lib/libkernel.so, host, include, CMakeLists.txt, ...}
    cmd = [
        "python3", _ppl_compile_py(),
        "--src", pl_path,
        "--chip", chip,
        "--mode", "pcie",
        "--rv",
        "--opt", opt,
        "--devid", str(devid),
        "--gen_test",
        "--out", workdir,
    ]
    if verbose:
        print("[ppl_runner] compile:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=workdir)

    kernel_so = os.path.join(workdir, "lib", "libkernel.so")
    if not os.path.isfile(kernel_so):
        raise RuntimeError(f"libkernel.so not produced at {kernel_so}")

    # 2) Append the ctypes wrapper target and build just that target.
    wrapper_src = f"{spec.kernel_name}_py_wrapper.cpp"
    wrapper_path = os.path.join(workdir, wrapper_src)
    with open(wrapper_path, "w") as f:
        f.write(_WRAPPER_TEMPLATE.format(kernel_name=spec.kernel_name))

    cmake_path = os.path.join(workdir, "CMakeLists.txt")
    with open(cmake_path, "a") as f:
        f.write(_wrapper_cmake_fragment(spec.kernel_name, wrapper_src))

    # Reuse the build tree ppl_compile.py created (workdir/build).  The appended
    # wrapper target is only picked up after a (re)configure: the Makefile
    # generator's auto-reconfigure-on-build is unreliable when the tree was
    # already configured by ppl_compile.py, so we reconfigure explicitly.
    # NB: pass NO -D flags here -- ppl_compile.py already baked the real chip
    # (e.g. tpub_7_1_e) and DEV_MODE into the cache; re-specifying -DCHIP would
    # override it and break the include paths.
    build_dir = os.path.join(workdir, "build")
    if not os.path.isdir(build_dir):
        os.makedirs(build_dir, exist_ok=True)
    subprocess.run(
        ["cmake", "-S", workdir, "-B", build_dir],
        check=True, cwd=workdir,
    )
    subprocess.run(
        ["cmake", "--build", build_dir, "--target", f"{spec.kernel_name}_py", "-j"],
        check=True, cwd=workdir,
    )
    wrapper_so = os.path.join(workdir, "lib", f"{spec.kernel_name}_py.so")
    if not os.path.isfile(wrapper_so):
        # fall back to the build tree install location
        cand = os.path.join(build_dir, f"{spec.kernel_name}_py.so")
        if os.path.isfile(cand):
            wrapper_so = cand
        else:
            raise RuntimeError(f"wrapper .so not found at {wrapper_so}")
    logger.warning(f"[TPU]: ppl_runner->build() done, kernel_so = {kernel_so}, wrapper_so = {wrapper_so}")
    return {"pl": pl_path, "kernel_so": kernel_so, "wrapper_so": wrapper_so, "workdir": workdir}


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
    """Populate the env vars the real SG2260E driver + npz_save require.

    Idempotent: only sets a var if the caller hasn't already.  Prepends the
    lib dirs the validated pcie recipe needs, in order:
      1. /opt/tpuv7/.../lib         -- the REAL driver (321KB) libtpuv7_rt.so;
                                       must come first so the deps/ emulator
                                       copy is not shadowed (it makes
                                       tpuRtInit exit 255 silently).
      2. deps/chip/tpub_7_1_e/lib    -- chip runtime (libcdm_* etc.)
      3. deps/runtime/tpuv7-runtime/lib -- device daemon / cmodel helpers
    """
    def _set(k: str, v: str) -> None:
        if not os.environ.get(k):
            os.environ[k] = v

    _set("TPU_OPER_PATH", _TPUV7_CURRENT + "/data")
    _set("AKS_MODULE_PATH", _TPUV7_CURRENT + "/data/AKS/libfirmware_core.so")
    _set("AKSV_MODULE_PATH", _TPUV7_CURRENT + "/data/AKSV/libfirmware_core.so")
    _set("PPL_TPUKERNEL_DEV_MODE", "pcie")
    _set("PPL_FILE_NAME", kernel_name)

    # Build the ordered list of lib dirs to prepend (those not already present).
    ppl_root = os.environ.get("PPL_PROJECT_ROOT", "")
    chip_lib = os.path.join(ppl_root, "deps", "chip", "tpub_7_1_e", "lib") if ppl_root else ""
    rt_lib = os.path.join(ppl_root, "deps", "runtime", "tpuv7-runtime", "lib") if ppl_root else ""
    ld_existing = os.environ.get("LD_LIBRARY_PATH", "")
    present = set(ld_existing.split(os.pathsep)) if ld_existing else set()
    to_prepend = [d for d in (_TPUV7_LIB, chip_lib, rt_lib) if d and d not in present]
    if to_prepend:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            to_prepend + ([ld_existing] if ld_existing else [])
        )


def _preload_real_driver() -> None:
    """Preload the REAL tpuv7 driver libs (RTLD_GLOBAL) before the wrapper .so.

    The wrapper's DT_RPATH points its ``libtpuv7_rt.so`` dependency at the
    *emulator* copy under ``deps/runtime/tpuv7-runtime/lib`` (which in turn
    needs ``libcdm_daemon_emulator.so`` and makes tpuRtInit exit 255).  Both
    the real and emulator copies share SONAME ``libtpuv7_rt.so``, so dlopen'ing
    the real one (321KB, from /opt/tpuv7/.../lib) with RTLD_GLOBAL *first* makes
    the linker reuse it for the wrapper's NEEDED entry -- the emulator (and its
    libcdm_daemon_emulator.so) is never loaded.  Setting LD_LIBRARY_PATH at
    runtime would be too late: ld.so only reads it at process startup.
    """
    for name in ("libtpuv7_rt.so", "libtpuv7_modelrt.so"):
        path = os.path.join(_TPUV7_LIB, name)
        if os.path.isfile(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass  # best-effort; the wrapper load will surface real errors


class PPLKernel:
    """ctypes handle to a built PPL GEMM(+ReLU) kernel on TPU."""

    def __init__(self, paths: dict[str, str], *, device: int = 3, kernel_name: str = "tl_gemm_relu"):
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
        L.py_dev_malloc.argtypes = [ctypes.c_uint64]
        L.py_dev_malloc.restype = ctypes.c_uint64
        L.py_dev_free.argtypes = [ctypes.c_uint64]
        L.py_memcpy_h2d.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64]
        L.py_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        launch = getattr(L, f"py_{kn}")
        print(type(launch))
        launch.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        launch.restype = ctypes.c_int
        self._launch = launch
        self.handle: Any = None

    def init(self) -> None:
        self.handle = self.lib.py_init_device(self.device)
        if not self.handle:
            raise RuntimeError("py_init_device failed (is the TPU device free?)")

    def close(self) -> None:
        if self.handle:
            self.lib.py_release_device(self.handle)
            self.handle = None

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *a):
        self.close()

    def _h2d(self, t: torch.Tensor) -> int:
        t = t.contiguous()
        addr = self.lib.py_dev_malloc(t.numel() * t.element_size())
        self.lib.py_memcpy_h2d(addr, ctypes.c_void_p(t.data_ptr()), t.numel() * t.element_size())
        return addr

    def _d2h(self, addr: int, t: torch.Tensor) -> None:
        t = t.contiguous()
        self.lib.py_memcpy_d2h(ctypes.c_void_p(t.data_ptr()), addr, t.numel() * t.element_size())

    def run(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """{fp16|bf16} [M,K] x [K,N] -> [M,N] (with ReLU if compiled with relu)."""
        logger.warning("[TPU]: PPLKernel->run()")
        assert a.dtype in (torch.float16, torch.bfloat16) and a.dtype == b.dtype
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        a_addr = self._h2d(a)
        b_addr = self._h2d(b)
        c = torch.empty((M, N), dtype=a.dtype)
        c_addr = self.lib.py_dev_malloc(c.numel() * c.element_size())
        try:
            if self._launch(self.handle, c_addr, a_addr, b_addr, M, K, N) != 0:
                raise RuntimeError("kernel launch failed")
            self.lib.py_sync_device(self.handle)
            self._d2h(c_addr, c)
        finally:
            self.lib.py_dev_free(a_addr)
            self.lib.py_dev_free(b_addr)
            self.lib.py_dev_free(c_addr)
        return c
