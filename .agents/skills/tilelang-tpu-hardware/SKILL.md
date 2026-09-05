---
name: tilelang-tpu-hardware
description: Use when working with Sophgo SG2260E TPU hardware — device setup, runtime environment, profiling, driver quirks, the PPL toolchain layout, and common runtime errors and their fixes.
---

# SG2260E TPU Hardware & Runtime

## Device Topology

- 8 TPU devices per machine: `/dev/sg-host-drv-0..7`
- **4 cores per device** (`MAX_TPU_CORE_NUM = 4` for `tpub_7_1_e`)
- PCIe mode (host-driven, not SoC mode)
- Chip arch identifier: `tpub_7_1_e` (used in PPL compile flags, LD_LIBRARY_PATH)
- Runtime variant suffix: `tpub_7_1_e_rv` (used in `--chip` flag and `_entry` symbols)

## Environment Setup

### Required Environment Variables

```bash
# PPL toolchain root
export PPL_PROJECT_ROOT=/workspace/ppl_v1.7.198-gcf5b037f-20260722

# TPU driver and firmware paths
source /opt/tpuv7/tpuv7-current/data/tpuv7-bin-path.sh
# ^ sets TPU_OPER_PATH, AKS_MODULE_PATH, AKSV_MODULE_PATH

# Library search path — ORDER MATTERS
export LD_LIBRARY_PATH=/opt/tpuv7/tpuv7-current/lib:$PPL_PROJECT_ROOT/deps/chip/tpub_7_1_e/lib:$PPL_PROJECT_ROOT/deps/runtime/tpuv7-runtime/lib:${LD_LIBRARY_PATH:-}

# PCIe mode
export PPL_TPUKERNEL_DEV_MODE=pcie
```

### Critical: LD_LIBRARY_PATH Order

The **real driver** `libtpuv7_rt.so` (~321KB) is at `/opt/tpuv7/tpuv7-current/lib/`
and MUST come first. The `deps/runtime/tpuv7-runtime/lib/libtpuv7_rt.so` is the big
**EMULATOR** (~50MB). Loading the emulator instead of the real driver → silent exit
255 with no error message.

### Device Selection

```bash
export TPU_VISIBLE_DEVICES=4  # use device 4 (0-indexed)
```

`tpu-smi` shows device status (needs the same `LD_LIBRARY_PATH`).

If a device is busy (held by another process), `tpuRtInit()` / `tpuRtSetDevice()`
will fail with `sgdev_communication: failed`. Switch to an idle device.

### Firmware Modules

`tpuv7-bin-path.sh` sets `AKS_MODULE_PATH` and `AKSV_MODULE_PATH` pointing to
firmware `.so` files the driver loads onto cores at init. Without these:
`basic_string: construction from null` error from `std::string(NULL)`.

## PPL Toolchain Layout

```
$PPL_PROJECT_ROOT/
  bin/ppl-compile              # PPL compiler binary (MLIR-based)
  python/tool/ppl_compile.py   # Python wrapper around ppl-compile
  deps/
    chip/tpub_7_1_e/lib/       # chip-specific libraries (libtpudnn.so, etc.)
    runtime/tpuv7-runtime/lib/ # EMULATOR runtime (NOT for real hardware)
    common/
      host/include/            # autotune.h, host_utils.h, npz_helper.h, ppl_mem.h
      dev/checker/             # device-side checker library
  inc/
    ppl.h                      # main PPL header
    ppl_tpu.h                  # core intrinsics: get_block_num(), set_block_num_max(), etc.
    ppl_defs.h                 # MULTI_CORE macro definition
  regression/                  # example .pl kernels (single and multi-core)
  examples/                    # Python and C++ examples
```

## Host API

### Device Lifecycle

```c
tpuRtInit()                    // init runtime
tpuRtSetDevice(devid)          // select device
tpuRtStreamCreate(&stream)     // create command stream
tpuRtKernelLoadModuleFile(path, stream)  // load libkernel.so
tpudnnHandleFromStream(devid, stream, module)  // create handle

// ... run kernels ...

tpudnnDestroy(handle)          // destroy handle
tpuRtKernelUnloadModule(module, stream)
tpuRtStreamSynchronize(stream)
tpuRtStreamDestroy(stream)
```

### Kernel Launch

```c
tpudnnLaunchKernel(handle, "kernel_entry", &api, sizeof(api), group_num, block_num)
```

- `group_num`: typically 1 (number of independent kernel groups)
- `block_num`: number of cores to use (1 = single-core, 4 = all cores on SG2260E)
- `tpudnnSync(handle)` blocks until kernel completes

### Memory Management

```c
tpuRtMalloc(&ptr, size, 1)    // device alloc
tpuRtFree(&ptr, 1)            // device free
tpuRtMemcpyS2D(dev_ptr, host_ptr, size)  // host → device
tpuRtMemcpyD2S(host_ptr, dev_ptr, size)  // device → host
```

## Profiling

### Enable / Disable

```c
// MUST set before tpuRtInit():
setenv("BMLIB_ENABLE_ALL_PROFILE", "1", 1);

tpudnnEnableProfile(handle, max_record_num, mode);
// ... run kernel ...
tpudnnSync(handle);
tpudnnDisableProfile(handle);  // MUST call before tpudnnDestroy()
```

**Critical constraints:**
- `BMLIB_ENABLE_ALL_PROFILE=1` must be set BEFORE `tpuRtInit()`. Setting it
  after has no effect. This means the runtime must be reinitialized if profiling
  is enabled after first use.
- `tpudnnDisableProfile()` MUST be called explicitly before `tpudnnDestroy()`.
  Omitting it causes a segfault in the destructor (`Profile1690::getProfileData()`).
- Do NOT change the process working directory between enable and disable.
  The TPU runtime writes `cdm_profile_data_dev*` to the CWD during disable;
  a CWD mismatch causes a segfault in `__fprintf_chk`.
- Default `max_record_num` of 4096 is too small for GEMM kernels. Use 131072+
  or read from `$PROFILE_RECORD_SIZE`.

### Post-Processing

```bash
bigTpuProfile <cdm_profile_data_devN-M> <output_dir>
# produces: summary.txt, perfetto.pftrace
```

Open `perfetto.pftrace` in https://ui.perfetto.dev/ for timeline visualization.

`bigTpuProfile` may fail with "profile timestamp normalization became negative"
on some devices — this is a tool/firmware compatibility issue, not a code bug.
The raw `cdm_profile_data_dev*` files are still valid for manual inspection.

### Profiling from Python (ctypes path)

```python
kernel.adapter.enable_profile(profiling_dir="/path/to/output")
c = kernel(a, b)  # run with profiling enabled
kernel.adapter.collect_profile()  # closes runtime, moves cdm files, runs bigTpuProfile
```

## Multi-Core

### PPL Device-Side API

```c
set_block_num_max();           // request all available cores
int n = get_block_num();       // how many cores were assigned
int i = get_block_index();     // this core's ID (0-based)
if (i >= n) return;            // guard for excess cores

// Alternative: explicit core count
set_block_num(4);
// Or use MULTI_CORE macro + template<int CoreNum>
```

See `ppl/inc/ppl_tpu.h` for the full list of core intrinsics.

### PPL Compiler Auto-Detection

The PPL compiler's `GroupBlockNumAssignPass` MLIR pass detects `set_block_num()`
or `set_block_num_max()` in the `.pl` source and automatically sets `block_num`
in the generated host code. No manual host code modification is needed.

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `sgdev_communication: failed` / `tpuRtKernelLoadModuleFile failed` | Device busy or wrong devid | Set `TPU_VISIBLE_DEVICES` to an idle device |
| Silent exit 255, no output | Loaded emulator instead of real driver | Fix `LD_LIBRARY_PATH` order: `/opt/tpuv7/tpuv7-current/lib` FIRST |
| `basic_string: construction from null` | Missing firmware module paths | Run `source tpuv7-bin-path.sh` |
| `std::logic_error` / exit 134 in npz_save | `PPL_FILE_NAME` not set | Set `PPL_FILE_NAME` env var |
| Segfault in `Profile1690::getProfileData()` during `tpudnnDestroy()` | Profile not disabled before destroy | Call `tpudnnDisableProfile()` before `tpudnnDestroy()` |
| Segfault in `__fprintf_chk` during `tpudnnDisableProfile()` | CWD changed between enable/disable | Never `os.chdir()` between profiling enable and disable |
| `Data overflow set max_record_num larger than 4096` | Default record buffer too small | Set `max_record_num` to 131072 or higher |
| `profile timestamp normalization became negative` | bigTpuProfile parsing issue | Not a code bug — tool/firmware compat issue. Raw data is still valid. |

## PPL Kernel Tile Body (GEMM Pattern)

The verified GEMM tile body uses 4D `[1,M,1,N]` layout with PPL's `gtensor` /
`tensor` / `sub_view` / `dma::load` / `dma::store` / `tiu::fmm2_nn`:

```c
dim4 res_max_shape = {1, block_m, 1, block_n};
auto sub_res = make_tensor<fp32>(res_max_shape, res_max_shape);
tiu::zero(sub_res);
for (int idx_k = 0; idx_k < K; idx_k += block_k) {
    enable_pipeline();
    dma::load(sub_left, left_gtensor.sub_view(left_max_shape, left_offset));
    dma::load(sub_right, right_gtensor.sub_view(right_max_shape, right_offset));
    bool last_k = (K - idx_k <= block_k);
    tiu::fmm2_nn(sub_res, sub_left, sub_right, bias, /*result_add=*/true,
                  DT_FP32, /*do_relu=*/(do_relu && last_k), saturate, requant);
}
tiu::cast(res_fp16, sub_res);
dma::store(res_gtensor.sub_view(res_max_shape, res_offset), res_fp16);
```

**do_relu semantics**: Gate `do_relu` on `last_k` (the final K iteration) or it
corrupts the fp32 accumulator by clamping intermediate partial sums to ≥0.
