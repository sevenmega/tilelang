from __future__ import annotations

from tvm.target import Target

from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def _detect_tpu_target() -> Target | str | None:
    return None


def normalize_tpu_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "tpu":
        return None
    return Target("tpu")


register_target_detector("tpu", _detect_tpu_target, override=True)
register_target_normalizer("tpu", normalize_tpu_target, override=True)
