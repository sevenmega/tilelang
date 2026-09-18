"""Generic TIR -> PPL translator for the tilelang TPU (SG2260E) backend.

This module walks a *lowered* tilelang ``PrimFunc`` (the output of
``JITImpl.get_tir`` / the trace of a ``@tilelang.jit(target="tpu")`` kernel) and
emits a PPL ``.pl`` kernel op-by-op -- there are **no per-family templates**.
The same translator handles GEMM, flash-attention, and any future kernel built
from the bounded op set below.

Node vocabulary handled (verified against the lowered GEMM and head-major
flash-attention TIR):

  * grid ``thread_binding`` For axes (blockIdx.x/y/z)  -> C ``for`` loops
  * ``tl.tileop.copy``   -> ``dma::load`` / ``dma::store`` / ``tiu::cast`` / ``tiu::move``
  * ``tl.tileop.fill``   -> ``tiu::zero`` / ``tiu::fill``
  * ``tl.tileop.gemm``   -> ``tiu::fmm2`` (fp16 operands, fp32 accum; result_add = !clear_accum)
  * ``tl.tileop.reduce`` -> ``quick_pooling`` (mode 0=max / 1=sum; +combine when !clear)
  * serial ``For``       -> inner C ``for`` loop
  * parallel ``For`` nest ending in a ``BufferStore`` -> ``tiu`` vector ops
        (``fadd``/``fsub``/``fmul``/``fmax``/``fmin``, scalar overloads,
         ``Div`` -> reciprocal + ``fmul``, ``exp`` -> ``exp_no_overflow``)

dim4 mapping (uniform, verified for 2D/3D/4D buffers):
  * local compute tile ``[r, c]`` -> ``{1, r, 1, c}``; ``[r]`` -> ``{1, r, 1, 1}``
    (rows on C, cols on W -- required for ``fmm2`` / ``quick_pooling`` /
    row-vector broadcast).
  * global buffer: pad its natural shape to 4D by *prepending* 1s and declare it
    with ``make_gtensor_permute<T>(mem_shape, GLOBAL, ptr, order)`` where
    ``order = {0, 2, 1, 3}`` -- this puts the trailing-two buffer axes onto C/W.
    A region's offset/extent are padded the same way and reordered by ``order``.

Shapes are baked as compile-time constants (concrete from the lowered TIR), one
compile per shape; the emitted ``__KERNEL__`` takes only pointers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# tilelang must be imported before tvm so the vendored TVM is on the path.
import tilelang  # noqa: F401
from tvm import tirx as _tir


# --------------------------------------------------------------------------- #
# dtype mapping                                                                #
# --------------------------------------------------------------------------- #

_TVM_TO_PPL = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}
_TVM_TO_TORCH = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}
_PPL_TO_DT = {"fp16": "DT_FP16", "bf16": "DT_BF16", "fp32": "DT_FP32"}


def _ppl_dtype(tvm_dtype: str) -> str:
    s = str(tvm_dtype)
    if s not in _TVM_TO_PPL:
        raise NotImplementedError(f"unsupported dtype for TPU codegen: {s!r}")
    return _TVM_TO_PPL[s]


# --------------------------------------------------------------------------- #
# kernel info returned to the runtime                                          #
# --------------------------------------------------------------------------- #


@dataclass
class BufArg:
    """One kernel pointer argument (a global buffer), in param order."""

    name: str            # buffer name (e.g. "Q", "C")
    ptr_name: str        # C pointer param name (e.g. "ptr_Q")
    ppl_dtype: str       # "fp16" / "bf16" / "fp32"
    torch_dtype: str     # "float16" / "bfloat16" / "float32"
    shape: tuple[int, ...]  # natural (row-major) buffer shape
    is_output: bool


@dataclass
class PPLKernelInfo:
    """Everything the runtime needs to bind + launch a translated kernel."""

    kernel_name: str
    args: list[BufArg]
    source: str = ""

    @property
    def inputs(self) -> list[BufArg]:
        return [a for a in self.args if not a.is_output]

    @property
    def outputs(self) -> list[BufArg]:
        return [a for a in self.args if a.is_output]


# --------------------------------------------------------------------------- #
# TIR expression rendering (offsets)                                           #
# --------------------------------------------------------------------------- #

_BIN_OP = {
    "Add": "+", "Sub": "-", "Mul": "*",
    "Div": "/", "FloorDiv": "/", "FloorMod": "%", "Mod": "%",
    "LT": "<", "LE": "<=", "GT": ">", "GE": ">=", "EQ": "==", "NE": "!=",
    "And": "&&", "Or": "||",
}


def _render_index(e: Any) -> str:
    """Render a TIR integer/boolean scalar expression to a C expression string."""
    tn = type(e).__name__
    if tn == "Var":
        return str(e.name)
    if tn in ("IntImm", "SizeVar"):
        try:
            return str(int(e))
        except (TypeError, ValueError):
            return str(e)
    if tn == "Not":
        return f"(!{_render_index(e.a)})"
    if tn in _BIN_OP:
        return f"({_render_index(e.a)} {_BIN_OP[tn]} {_render_index(e.b)})"
    # Fall back to the TVM printer (e.g. constant-folded scalars).
    try:
        return str(int(e))
    except (TypeError, ValueError):
        raise NotImplementedError(f"cannot render index expr {tn}: {e}")


def _is_zero(e: Any) -> bool:
    try:
        return int(e) == 0
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# region parsing                                                               #
# --------------------------------------------------------------------------- #


@dataclass
class Region:
    buffer: Any
    indices: list[Any]
    mask: int
    extents: list[Any]


def _parse_region(call: Any) -> Region:
    """Parse a ``tl.tileop.region`` Call -> (buffer, indices, mask, extents)."""
    load = call.args[0]           # BufferLoad(buf[indices...])
    mask = int(call.args[1])
    extents = list(call.args[2:])
    return Region(load.buffer, list(load.indices), mask, extents)


def _op_name(call: Any) -> str:
    return call.op.name


# --------------------------------------------------------------------------- #
# emitter                                                                      #
# --------------------------------------------------------------------------- #


class PPLEmitter:
    """Walks a lowered PrimFunc body and accumulates PPL source lines."""

    ORDER = (0, 2, 1, 3)

    def __init__(self, func: Any, kernel_name: str):
        self.func = func
        self.kernel_name = kernel_name
        self.lines: list[str] = []
        self._tmp = 0

        # Classify buffers: globals (param buffer_map) vs locals (alloc_buffers).
        self.globals: dict[str, BufArg] = {}
        self.global_bufs: dict[str, Any] = {}
        self.locals: dict[str, tuple[tuple[int, int, int, int], str]] = {}

        out_idx = [int(i) for i in func.attrs["tilelang_out_idx"]]
        params = list(func.params)
        n = len(params)
        out_set = {i % n for i in out_idx}

        self.args: list[BufArg] = []
        for i, p in enumerate(params):
            buf = func.buffer_map[p]
            ppl_dt = _ppl_dtype(buf.dtype)
            shape = tuple(int(s) for s in buf.shape)
            arg = BufArg(
                name=buf.name,
                ptr_name=f"ptr_{buf.name}",
                ppl_dtype=ppl_dt,
                torch_dtype=_TVM_TO_TORCH[str(buf.dtype)],
                shape=shape,
                is_output=(i in out_set),
            )
            self.args.append(arg)
            self.globals[buf.name] = arg
            self.global_bufs[buf.name] = buf

    # -- naming helpers ---------------------------------------------------- #

    def _new_tmp(self, prefix: str = "t") -> str:
        self._tmp += 1
        return f"{prefix}{self._tmp}"

    def emit(self, line: str, indent: int = 1) -> None:
        self.lines.append("  " * indent + line)

    # -- dim4 helpers ------------------------------------------------------ #

    @staticmethod
    def _tile_dim4(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Local compute tile [r,c] -> {1,r,1,c}; [r] -> {1,r,1,1}."""
        if len(shape) == 2:
            return (1, shape[0], 1, shape[1])
        if len(shape) == 1:
            return (1, shape[0], 1, 1)
        raise NotImplementedError(f"local tile rank {len(shape)} not supported: {shape}")

    def tile_dim4(self, name: str) -> tuple[int, int, int, int]:
        return self.locals[name][0]

    @staticmethod
    def _dim4_str(d: tuple[int, int, int, int]) -> str:
        return "{" + ", ".join(str(x) for x in d) + "}"

    def _pad4_prepend(self, seq: list[Any], fill: Any) -> list[Any]:
        seq = list(seq)
        while len(seq) < 4:
            seq.insert(0, fill)
        if len(seq) > 4:
            raise NotImplementedError(f"buffer rank >4 not supported: {seq}")
        return seq

    def _permute4(self, padded: list[Any]) -> list[Any]:
        return [padded[o] for o in self.ORDER]

    # -- global tensor declarations ---------------------------------------- #

    def emit_global_decls(self) -> None:
        self.emit("int order[4] = {0, 2, 1, 3};")
        for arg in self.args:
            mem = self._pad4_prepend([str(s) for s in arg.shape], "1")
            self.emit(f"dim4 {arg.name}_mem_shape = {{{', '.join(mem)}}};")
            self.emit(
                f"auto {arg.name}_gt = make_gtensor_permute<{arg.ppl_dtype}>("
                f"{arg.name}_mem_shape, GLOBAL, {arg.ptr_name}, order);"
            )

    # -- local allocations ------------------------------------------------- #

    def emit_local_allocs(self, alloc_buffers: list[Any], indent: int) -> None:
        for buf in alloc_buffers:
            shape = tuple(int(s) for s in buf.shape)
            d4 = self._tile_dim4(shape)
            dt = _ppl_dtype(buf.dtype)
            self.locals[buf.name] = (d4, dt)
            s = self._dim4_str(d4)
            self.emit(f"dim4 {buf.name}_bs = {s};", indent)
            self.emit(
                f"auto {buf.name} = make_tensor<{dt}>({buf.name}_bs, {buf.name}_bs);",
                indent,
            )

    # -- region -> global sub_view / local tile --------------------------- #

    def _is_global(self, buf: Any) -> bool:
        return buf.name in self.globals and buf.name not in self.locals

    def _subview(self, reg: Region) -> str:
        """Emit dim4 shape/offset for a global region; return sub_view expr."""
        arg = self.globals[reg.buffer.name]
        padoff = self._pad4_prepend([_render_index(i) for i in reg.indices], "0")
        padext = self._pad4_prepend([_render_index(e) for e in reg.extents], "1")
        loff = self._permute4(padoff)
        lext = self._permute4(padext)
        off_name = self._new_tmp("off")
        shp_name = self._new_tmp("shp")
        self.emit(f"dim4 {shp_name} = {{{', '.join(lext)}}};")
        self.emit(f"dim4 {off_name} = {{{', '.join(loff)}}};")
        return f"{reg.buffer.name}_gt.sub_view({shp_name}, {off_name})"

    # -- statement dispatch ------------------------------------------------ #

    def emit_stmt(self, node: Any) -> None:
        tn = type(node).__name__
        if tn == "Evaluate":
            self.emit_evaluate(node.value)
        elif tn == "For":
            self.emit_for(node)
        elif tn == "SeqStmt":
            for s in node.seq:
                self.emit_stmt(s)
        elif tn in ("SBlockRealize", "BlockRealize"):
            self.emit_stmt(node.block)
        elif tn in ("SBlock", "Block"):
            self.emit_stmt(node.body)
        elif tn == "BufferStore":
            self.emit_store(node)
        elif tn == "IfThenElse":
            cond = _render_index(node.condition)
            self.emit(f"if ({cond}) {{")
            self.emit_stmt(node.then_case)
            if getattr(node, "else_case", None) is not None:
                self.emit("} else {")
                self.emit_stmt(node.else_case)
            self.emit("}")
        else:
            raise NotImplementedError(f"unhandled statement node: {tn}")

    def emit_evaluate(self, call: Any) -> None:
        if type(call).__name__ != "Call":
            raise NotImplementedError(f"unhandled Evaluate value: {type(call).__name__}")
        op = _op_name(call)
        if op == "tl.tileop.copy":
            self.emit_copy(call)
        elif op == "tl.tileop.fill":
            self.emit_fill(call)
        elif op == "tl.tileop.gemm":
            self.emit_gemm(call)
        elif op == "tl.tileop.reduce":
            self.emit_reduce(call)
        else:
            raise NotImplementedError(f"unhandled tile op: {op}")

    # -- copy -------------------------------------------------------------- #

    def emit_copy(self, call: Any) -> None:
        src = _parse_region(call.args[0])
        dst = _parse_region(call.args[1])
        src_g = self._is_global(src.buffer)
        dst_g = self._is_global(dst.buffer)
        if src_g and not dst_g:
            self.emit(f"// copy {src.buffer.name} -> {dst.buffer.name} (load)")
            view = self._subview(src)
            self.emit(f"dma::load({dst.buffer.name}, {view});")
        elif dst_g and not src_g:
            self.emit(f"// copy {src.buffer.name} -> {dst.buffer.name} (store)")
            view = self._subview(dst)
            src_dt = self.locals[src.buffer.name][1]
            dst_dt = self.globals[dst.buffer.name].ppl_dtype
            store_src = src.buffer.name
            if src_dt != dst_dt:
                # dma::store requires the on-chip tile to match the global dtype;
                # cast the (fp32) accumulator into a dst-dtype companion first.
                tmp = self._new_tmp("st")
                bs = self._new_tmp("stbs")
                shape = self.tile_dim4(src.buffer.name)
                self.emit(f"dim4 {bs} = {self._dim4_str(shape)};")
                self.emit(f"auto {tmp} = make_tensor<{dst_dt}>({bs}, {bs});")
                self.emit(f"tiu::cast({tmp}, {src.buffer.name});")
                store_src = tmp
            self.emit(f"dma::store({view}, {store_src});")
        elif not src_g and not dst_g:
            src_dt = self.locals[src.buffer.name][1]
            dst_dt = self.locals[dst.buffer.name][1]
            if src_dt == dst_dt:
                self.emit(f"tiu::move({dst.buffer.name}, {src.buffer.name});")
            else:
                self.emit(f"tiu::cast({dst.buffer.name}, {src.buffer.name});")
        else:
            raise NotImplementedError("global->global copy not supported")

    # -- fill -------------------------------------------------------------- #

    def emit_fill(self, call: Any) -> None:
        dst = _parse_region(call.args[0])
        val = call.args[1]
        name = dst.buffer.name
        if _is_zero(val):
            self.emit(f"tiu::zero({name});")
        else:
            self.emit(f"tiu::fill({name}, {float(val)});")

    # -- gemm -------------------------------------------------------------- #

    def emit_gemm(self, call: Any) -> None:
        a = _parse_region(call.args[0])
        b = _parse_region(call.args[1])
        c = _parse_region(call.args[2])
        ltrans = bool(int(call.args[3]))
        rtrans = bool(int(call.args[4]))
        clear_accum = bool(int(call.args[9]))
        result_add = "false" if clear_accum else "true"
        out_dt = _PPL_TO_DT[self.locals[c.buffer.name][1]]
        self.emit(
            f"tiu::fmm2({c.buffer.name}, {a.buffer.name}, {b.buffer.name}, "
            f"{'true' if ltrans else 'false'}, {'true' if rtrans else 'false'}, "
            f"false, false, {result_add}, {out_dt});"
        )

    # -- reduce ------------------------------------------------------------ #

    def emit_reduce(self, call: Any) -> None:
        src = _parse_region(call.args[0])
        dst = _parse_region(call.args[1])
        mode_str = str(call.args[2]).strip('"')
        dim = int(call.args[3])
        clear = bool(int(call.args[4]))
        src_shape = tuple(int(s) for s in src.buffer.shape)
        if dim != len(src_shape) - 1:
            raise NotImplementedError(
                f"reduce dim {dim} must be the last axis of {src_shape}")
        mode = 0 if mode_str == "max" else 1
        fill = "-30000.0" if mode_str == "max" else "0.0"
        d4 = self._tile_dim4(src_shape)
        bs = self._new_tmp("rbs")
        self.emit(f"dim4 {bs} = {self._dim4_str(d4)};")
        if clear:
            self.emit(
                f"quick_pooling({dst.buffer.name}, {src.buffer.name}, "
                f"&{bs}, &{bs}, {fill}, {mode});")
        else:
            dst_dt = self.locals[dst.buffer.name][1]
            dst_d4 = self.tile_dim4(dst.buffer.name)
            tmp = self._new_tmp("red")
            tbs = self._new_tmp("rdbs")
            self.emit(f"dim4 {tbs} = {self._dim4_str(dst_d4)};")
            self.emit(f"auto {tmp} = make_tensor<{dst_dt}>({tbs}, {tbs});")
            self.emit(
                f"quick_pooling({tmp}, {src.buffer.name}, "
                f"&{bs}, &{bs}, {fill}, {mode});")
            combine = "tiu::fmax" if mode == 0 else "tiu::fadd"
            self.emit(f"{combine}({dst.buffer.name}, {dst.buffer.name}, {tmp});")

    # -- parallel elementwise loops ---------------------------------------- #

    def emit_for(self, node: Any) -> None:
        kind = int(node.kind)
        if kind == 1:  # parallel -> elementwise tile op
            store = self._innermost_store(node)
            self.emit_store(store)
        elif kind in (0, 3):  # serial / unrolled -> C for loop
            var = node.loop_var.name
            ext = _render_index(node.extent)
            self.emit(f"for (int {var} = 0; {var} < {ext}; {var}++) {{")
            self.emit_stmt(node.body)
            self.emit("}")
        else:
            raise NotImplementedError(f"For kind {kind} not supported outside grid")

    @staticmethod
    def _innermost_store(node: Any) -> Any:
        while type(node).__name__ == "For":
            node = node.body
        if type(node).__name__ != "BufferStore":
            raise NotImplementedError(
                f"parallel loop body must be a BufferStore, got {type(node).__name__}")
        return node

    # -- expression emitter ------------------------------------------------ #

    def _expr_shape(self, e: Any) -> tuple[int, int, int, int] | None:
        tn = type(e).__name__
        if tn == "BufferLoad":
            return self.tile_dim4(e.buffer.name)
        if tn in ("FloatImm", "IntImm"):
            return None
        if tn == "Call" and _op_name(e).endswith(".exp"):
            return self._expr_shape(e.args[0])
        if tn in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
            sa = self._expr_shape(e.a)
            sb = self._expr_shape(e.b)
            if sa is None:
                return sb
            if sb is None:
                return sa
            return tuple(max(x, y) for x, y in zip(sa, sb))
        raise NotImplementedError(f"cannot infer shape of expr {tn}")

    def _scalar_val(self, e: Any) -> str | None:
        tn = type(e).__name__
        if tn in ("FloatImm", "IntImm"):
            return repr(float(e))
        return None

    def _gen_value(self, e: Any, dtype: str):
        """Return ('scalar', str) or ('tensor', varname)."""
        s = self._scalar_val(e)
        if s is not None:
            return ("scalar", s)
        tn = type(e).__name__
        if tn == "BufferLoad":
            return ("tensor", e.buffer.name)
        # nested expression -> materialize into a temp
        shape = self._expr_shape(e)
        tmp = self._new_tmp("e")
        bs = self._new_tmp("ebs")
        self.emit(f"dim4 {bs} = {self._dim4_str(shape)};")
        self.emit(f"auto {tmp} = make_tensor<{dtype}>({bs}, {bs});")
        self._emit_into(e, tmp, dtype)
        return ("tensor", tmp)

    def emit_store(self, store: Any) -> None:
        into = store.buffer.name
        dtype = self.locals[into][1]
        self._emit_into(store.value, into, dtype)

    def _emit_into(self, e: Any, into: str, dtype: str) -> None:
        tn = type(e).__name__
        if tn == "Call" and _op_name(e).endswith(".exp"):
            arg = self._gen_value(e.args[0], dtype)
            assert arg[0] == "tensor", "exp of a scalar is unexpected"
            shape = self._expr_shape(e.args[0])
            bs = self._new_tmp("xbs")
            self.emit(f"dim4 {bs} = {self._dim4_str(shape)};")
            self.emit(f"exp_no_overflow({into}, {arg[1]}, &{bs}, &{bs});")
            return
        if tn in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
            self._emit_binary(tn, e.a, e.b, into, dtype)
            return
        if tn == "BufferLoad":
            self.emit(f"tiu::move({into}, {e.buffer.name});")
            return
        s = self._scalar_val(e)
        if s is not None:
            self.emit(f"tiu::fill({into}, {s});")
            return
        raise NotImplementedError(f"cannot emit expression {tn}")

    def _emit_binary(self, opn: str, a: Any, b: Any, into: str, dtype: str) -> None:
        av = self._gen_value(a, dtype)
        bv = self._gen_value(b, dtype)
        a_scalar = av[0] == "scalar"
        b_scalar = bv[0] == "scalar"

        if opn == "Div":
            if b_scalar:
                # a / C  ->  a * (1/C)
                inv = 1.0 / float(bv[1])
                self.emit(f"tiu::fmul({into}, {av[1]}, {repr(inv)});")
                return
            if a_scalar:
                self.emit(f"tiu::fdiv({into}, {av[1]}, {bv[1]}, 3);")
                return
            # tensor / tensor -> reciprocal of denominator, then multiply
            shape = self._expr_shape(b)
            recip = self._new_tmp("rc")
            bs = self._new_tmp("rcbs")
            self.emit(f"dim4 {bs} = {self._dim4_str(shape)};")
            self.emit(f"auto {recip} = make_tensor<{dtype}>({bs}, {bs});")
            self.emit(f"tiu::fdiv({recip}, 1.0f, {bv[1]}, 3);")
            self.emit(f"tiu::fmul({into}, {av[1]}, {recip});")
            return

        fn = {"Add": "fadd", "Sub": "fsub", "Mul": "fmul",
              "Max": "fmax", "Min": "fmin"}[opn]

        if a_scalar and b_scalar:
            raise NotImplementedError("binary op of two scalars should be folded")

        if not a_scalar and not b_scalar:
            self.emit(f"tiu::{fn}({into}, {av[1]}, {bv[1]});")
            return

        # one scalar operand
        if opn in ("Add", "Mul", "Max", "Min"):  # commutative
            tensor = av[1] if not a_scalar else bv[1]
            scalar = bv[1] if not a_scalar else av[1]
            self.emit(f"tiu::{fn}({into}, {tensor}, {scalar});")
        elif opn == "Sub":
            if b_scalar:  # a - C
                self.emit(f"tiu::fsub({into}, {av[1]}, {bv[1]});")
            else:         # C - b
                self.emit(f"tiu::fsub({into}, {av[1]}, {bv[1]});")
        else:
            raise NotImplementedError(f"binary {opn} with scalar")

    # -- top level --------------------------------------------------------- #

    def _walk_to_kernel(self) -> tuple[list[Any], list[Any], Any]:
        """Return (grid_fors, alloc_buffers, body) from the lowered func body."""
        node = self.func.body
        grid: list[Any] = []
        # unwrap the outer root block(s)
        while True:
            tn = type(node).__name__
            if tn in ("SBlockRealize", "BlockRealize"):
                node = node.block
            elif tn in ("SBlock", "Block") and not grid:
                node = node.body
            elif tn == "For" and int(node.kind) == 4:
                grid.append(node)
                node = node.body
            else:
                break
        # node is now the innermost SBlockRealize / SBlock (tilelang_root)
        while type(node).__name__ in ("SBlockRealize", "BlockRealize"):
            node = node.block
        if type(node).__name__ not in ("SBlock", "Block"):
            raise NotImplementedError(f"expected kernel block, got {type(node).__name__}")
        allocs = list(getattr(node, "alloc_buffers", []) or [])
        return grid, allocs, node.body

    def run(self) -> PPLKernelInfo:
        grid, allocs, body = self._walk_to_kernel()

        # kernel signature
        sig = ", ".join(f"{a.ppl_dtype} *{a.ptr_name}" for a in self.args)
        self.lines.append('#include "ppl.h"')
        self.lines.append('#include "ppl_wrapper_func.h"')
        self.lines.append("")
        self.lines.append("using namespace ppl;")
        self.lines.append("")
        self.lines.append(f"__KERNEL__ void {self.kernel_name}({sig}) {{")

        self.emit_global_decls()

        # grid loops
        indent_base = 1
        for g in grid:
            var = g.loop_var.name
            ext = _render_index(g.extent)
            self.emit(f"for (int {var} = 0; {var} < {ext}; {var}++) {{", indent_base)
            indent_base += 1

        # NOTE: from here we rely on self.emit's default indent (1) for body
        # readability; PPL/C++ does not care about indentation.
        self.emit_local_allocs(allocs, indent_base)
        self.emit_stmt(body)

        # close grid loops
        for g in reversed(grid):
            indent_base -= 1
            self.emit("}", indent_base)
        self.lines.append("}")

        # __TEST__ stub
        self.lines.append("")
        self._emit_test_stub()

        source = "\n".join(self.lines) + "\n"
        return PPLKernelInfo(kernel_name=self.kernel_name, args=self.args, source=source)

    def _emit_test_stub(self) -> None:
        self.lines.append(f"__TEST__ void {self.kernel_name}_main() {{")
        call_args = []
        for a in self.args:
            mem = self._pad4_prepend([str(s) for s in a.shape], "1")
            self.lines.append(
                f"  dim4 {a.name}_ts = {{{', '.join(mem)}}};")
            self.lines.append(
                f"  {a.ppl_dtype} *{a.ptr_name} = ppl::malloc<{a.ppl_dtype}>(&{a.name}_ts);")
            if not a.is_output:
                self.lines.append(f"  ppl::rand({a.ptr_name}, &{a.name}_ts, -1.0, 1.0);")
            call_args.append(a.ptr_name)
        self.lines.append(f"  {self.kernel_name}({', '.join(call_args)});")
        self.lines.append("}")


def translate(func: Any, kernel_name: str | None = None) -> PPLKernelInfo:
    """Translate a lowered tilelang PrimFunc into a PPL kernel + arg metadata."""
    if kernel_name is None:
        kernel_name = str(func.attrs.get("global_symbol", "tl_kernel"))
    return PPLEmitter(func, kernel_name).run()
