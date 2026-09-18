"""Reusable ``@tilelang.jit(target="tpu")`` DSL kernels for the PPL backend.

These are *real* tilelang DSL kernels (no PPL templates).  They lower to TIR and
are compiled to PPL by the generic TIR->PPL translator (``tilelang.tpu.ppl_codegen``)
through the generic ``tilelang.tpu.compile`` path.  Import the kernel and call
``.compile(**dims)`` (or ``.get_tir(**dims)``) to instantiate for concrete shapes.
"""

from .attention import flash_attention_gqa

__all__ = ["flash_attention_gqa"]
