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
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# PPL root resolution                                                          #
# --------------------------------------------------------------------------- #

def get_tpu_device() -> int:
    """Return the TPU device ID from ``$TPU_VISIBLE_DEVICES`` (default 0)."""
    val = os.environ.get("TPU_VISIBLE_DEVICES", "0")
    try:
        return int(val.split(",")[0])
    except (ValueError, IndexError):
        return 0


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


def emit_pl(spec: PPLGemmSpec) -> str:
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


def build(
    spec: PPLGemmSpec,
    workdir: str,
    *,
    chip: str = "sg2260e",
    opt: str = "O3",
    verbose: bool = False,
) -> dict[str, str]:
    """Emit .pl, compile device code, cmake-build libkernel.so + ctypes wrapper + test_case.

    Always compiles with ``--autotune`` so the ``test_case`` binary includes
    profiling instrumentation.  Profiling data can be collected later via
    ``run_profiling()`` without recompilation.

    Uses the inline path (calling ``ppl-compile`` binary + cmake directly) when
    the ``ppl-compile`` binary is found.  Falls back to shelling out to
    ``ppl_compile.py`` as a subprocess otherwise (profiling unavailable in
    fallback mode).

    Compilation is device-agnostic — no device ID is needed here.

    Returns a dict with paths:
        {"pl", "kernel_so", "wrapper_so", "test_case", "workdir"}.
    ``test_case`` may be ``None`` when built via the fallback path.
    """
    logger.warning("[TPU]: ppl_runner->build()")
    ppl_root = _get_ppl_root()
    chip_arch = _resolve_chip_arch(ppl_root, chip)

    os.makedirs(workdir, exist_ok=True)
    pl_content = emit_pl(spec)
    pl_path = os.path.join(workdir, f"{spec.kernel_name}.pl")
    kernel_so = os.path.join(workdir, "lib", "libkernel.so")
    wrapper_so = os.path.join(workdir, "lib", f"{spec.kernel_name}_py.so")
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
        _setup_ppl_env(ppl_root, chip, chip_arch, workdir, spec.kernel_name)
        _run_ppl_compile(ppl_root, pl_path, chip_arch, workdir, opt=opt, rv=True,
                         autotune=True, verbose=verbose)
        _cmake_build(ppl_root, chip_arch, workdir, mode="pcie", verbose=verbose)
    else:
        logger.warning("[TPU]: ppl-compile binary not found, falling back to ppl_compile.py subprocess")
        _build_via_ppl_compile_py(ppl_root, pl_path, workdir, chip, opt, verbose)

    if not os.path.isfile(kernel_so):
        raise RuntimeError(f"libkernel.so not produced at {kernel_so}")

    wrapper_src = f"{spec.kernel_name}_py_wrapper.cpp"
    wrapper_path = os.path.join(workdir, wrapper_src)
    with open(wrapper_path, "w") as f:
        f.write(_WRAPPER_TEMPLATE.format(kernel_name=spec.kernel_name))

    cmake_path = os.path.join(workdir, "CMakeLists.txt")
    with open(cmake_path, "a") as f:
        f.write(_wrapper_cmake_fragment(spec.kernel_name, wrapper_src))

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


def _collect_profile_data(profiling_dir: str, *, verbose: bool = False) -> dict[str, Any]:
    """Process ``cdm_profile_data_dev*`` files with ``bigTpuProfile``.

    If ``bigTpuProfile`` fails (e.g. timestamp normalization errors), the raw
    ``cdm_profile_data_dev*`` files are still returned so the user can inspect
    or reprocess them manually.  A warning is printed instead of raising.
    """
    import glob as _glob
    import sys as _sys
    import warnings as _warnings

    cdm_files = sorted(_glob.glob(os.path.join(profiling_dir, "cdm_profile_data_dev*")))
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

    return _collect_profile_data(profiling_dir, verbose=verbose)


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
        L.py_enable_profile.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.py_enable_profile.restype = ctypes.c_int
        L.py_disable_profile.argtypes = [ctypes.c_void_p]
        L.py_disable_profile.restype = ctypes.c_int
        self.handle: Any = None
        self._profile_dir: str | None = None
        self._profile_active: bool = False

    def init(self) -> None:
        self.handle = self.lib.py_init_device(self.device)
        if not self.handle:
            raise RuntimeError("py_init_device failed (is the TPU device free?)")

    def close(self) -> None:
        if self.handle:
            if self._profile_active:
                self.lib.py_disable_profile(self.handle)
                self._profile_active = False
                self._move_profile_data()
            self.lib.py_release_device(self.handle)
            self.handle = None

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
        ret = self.lib.py_enable_profile(self.handle, max_record_num, book_keeping)
        if ret != 0:
            raise RuntimeError(f"tpudnnEnableProfile failed with code {ret}")
        self._profile_active = True

    def collect_profile(self, *, verbose: bool = False) -> dict[str, Any]:
        """Process ``cdm_profile_data_dev*`` files with ``bigTpuProfile``."""
        if not self._profile_dir:
            raise RuntimeError("Profiling not enabled — no profile_dir set")
        return _collect_profile_data(self._profile_dir, verbose=verbose)

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
