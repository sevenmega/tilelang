#include <tvm/target/target_kind.h>

namespace tvm {

TVM_REGISTER_TARGET_KIND("tpu", kDLCPU)
    .add_attr_option<ffi::String>("mcpu")
    .set_default_keys({"cpu"});

}  // namespace tvm
