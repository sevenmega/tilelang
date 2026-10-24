"""``gemm`` for TPU tiles that carry an interior unit axis.

The generic :func:`tilelang.language.gemm` requires every axis of an operand
except the trailing two to be extent 1.  A TPU kernel that copies a ``[rows, D]``
slab out of a *3-D* global buffer declares the on-chip tile ``(rows, 1, D)`` so
the copy region's extents line up with the buffer's axes -- ``Q`` is
``[T, Hq, D]`` and the slab is ``Q[r:r+rows, h, :]`` (see
``tilelang.tpu.kernels.attention_paged.flash_attention_paged``).  On chip that tile is
the plain ``rows x D`` matrix: its ``dim4`` is ``{1, rows, 1, D}``, exactly what a
2-D ``(rows, D)`` tile produces, and the PPL lowering emits ``tiu::fmm2``
straight from the tile descriptors without reading M/N/K.  The frontend check is
therefore pure formality here, and this variant derives the matrix axes from the
leading and trailing axes instead of rejecting the shape.
"""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language.utils import buffer_region_to_tile_region
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import (
    prim_expr_equal,
    retrieve_shape,
    to_buffer_region,
)


def _matrix_shape(shape: list) -> list:
    """The two matrix axes of a tile, dropping an interior unit axis."""
    if len(shape) == 3 and int(shape[1]) == 1:
        return [shape[0], shape[2]]
    return list(shape[-2:])


def gemm_tiled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
) -> tirx.PrimExpr:
    """``T.gemm`` that accepts a ``(rows, 1, K)`` / ``(rows, D)`` operand pair."""
    A_region = to_buffer_region(A)
    B_region = to_buffer_region(B)
    C_region = to_buffer_region(C)
    A_shape = _matrix_shape(retrieve_shape(A_region))
    B_shape = _matrix_shape(retrieve_shape(B_region))
    C_shape = _matrix_shape(retrieve_shape(C_region))

    M, N = C_shape[-2], C_shape[-1]
    M_A = A_shape[-1] if transpose_A else A_shape[-2]
    K = A_shape[-2] if transpose_A else A_shape[-1]
    N_B = B_shape[-2] if transpose_B else B_shape[-1]
    K_B = B_shape[-1] if transpose_B else B_shape[-2]
    assert prim_expr_equal(M_A, M), f"gemm M shape check failed: M_A = {M_A}, M_C = {M}"
    assert prim_expr_equal(K, K_B), f"gemm K shape check failed: K_A = {K}, K_B = {K_B}"
    assert prim_expr_equal(N_B, N), f"gemm N shape check failed: N_B = {N_B}, N_C = {N}"

    A_arg = buffer_region_to_tile_region(A_region, "r", list(retrieve_shape(A_region)))
    B_arg = buffer_region_to_tile_region(B_region, "r", list(retrieve_shape(B_region)))
    C_arg = buffer_region_to_tile_region(C_region, "rw", list(retrieve_shape(C_region)))
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.gemm"),
        A_arg,
        B_arg,
        C_arg,
        transpose_A,
        transpose_B,
        M,
        N,
        K,
        policy,
        clear_accum,
        0,  # stride_a / stride_b / offset_a / offset_b: unread by the TPU lowering
        0,
        0,
        0,
        1,  # k_pack
        0,  # wg_wait
        tirx.const(0, dtype="int32"),  # mbar (unused on TPU)
        0,
        0,
    )
