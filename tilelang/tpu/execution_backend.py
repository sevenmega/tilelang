from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend

register_execution_backend(
    "tpu",
    ExecutionBackendSpec("ppl", enable_host_codegen=False, enable_device_compile=False),
    override=True,
)
