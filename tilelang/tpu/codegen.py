from __future__ import annotations

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen
from tilelang.backend.host_codegen import HostCodegen, global_func_host_codegen, register_host_codegen

register_device_codegen(
    "tpu",
    DeviceCodegen(
        "tpu",
        build_without_compile=global_func_device_codegen("target.build.tilelang_c"),
    ),
    override=True,
)

register_host_codegen(
    "tpu",
    HostCodegen("tpu", build=global_func_host_codegen("target.build.tilelang_c_host")),
    override=True,
)
