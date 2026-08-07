/*!
 * \file src/tpu/target_kind.cc
 * \brief Register the "tpu" target kind (SG2260E TPU via the PPL toolchain).
 *
 * The TPU code path lowers tilelang kernels to a PPL ``.pl`` program (see
 * :mod:`tilelang.tpu.ppl_runner`) instead of going through a TVM codegen, so
 * this target kind is a thin registration so that ``target="tpu"`` is a valid
 * TVM target string; no LLVM/C backend is involved.
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>

namespace tvm {

TVM_REGISTER_TARGET_KIND("tpu", kDLCPU)
    .add_attr_option<ffi::String>("mcpu")
    .set_default_keys({"cpu"});

}  // namespace tvm
