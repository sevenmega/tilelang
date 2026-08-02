from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from tilelang.cpu.op.gemm.gemm_scalar import GEMM_INST_SCALAR, GemmScalar


def _match_tpu(target) -> bool:
    return target.kind.name == "tpu"


register_gemm_impl("tpu.scalar", GEMM_INST_SCALAR, _match_tpu, GemmScalar)
