"""Generic TIR -> PPL translator for the tilelang TPU (SG2260E) backend.

This module walks a *lowered* tilelang ``PrimFunc`` (the output of
``JITImpl.get_tir`` / the trace of a ``@tilelang.jit(target="tpu")`` kernel) and
emits a PPL ``.pl`` kernel op-by-op -- there are **no per-family templates**.
The same translator handles GEMM, flash-attention, and any future kernel built
from the bounded op set below.

Node vocabulary handled (verified against the lowered GEMM and head-major
flash-attention TIR):

  * grid ``thread_binding`` For axes (blockIdx.x/y/z).  Grid *arity* selects the
    dispatch: a 2-axis grid (no core axis) becomes plain nested C ``for``
    loops, while a 3-axis grid is multi-core -- the last axis is the core index,
    emitted as ``get_block_index()`` behind a ``get_block_num()`` guard instead
    of a loop, so the body runs once per core rather than serially on one core.
  * ``tl.tileop.copy``   -> ``dma::load`` / ``dma::store`` / ``tiu::cast`` / ``tiu::move``
  * ``tl.tileop.fill``   -> ``tiu::zero`` / ``tiu::fill``
  * ``tl.tileop.gemm``   -> ``tiu::fmm2`` (fp16 operands, fp32 accum; result_add = !clear_accum)
  * ``tl.tileop.reduce`` -> ``quick_pooling`` (mode 0=max / 1=sum; +combine when !clear)
  * serial ``For``       -> inner C ``for`` loop.  A ``num_stages`` annotation
        (i.e. the loop came from ``T.Pipelined``) additionally emits
        ``enable_pipeline()`` as the loop body's first statement, handing the
        double/triple-buffering and load/compute overlap to the PPL compiler.
        A plain ``T.serial`` loop gets no hint and runs straight through.
  * ``Bind`` (eager-builder ``let``, e.g. ``bx_global = bc * M_tiles + bx``)
        -> ``int <var> = <value>;`` at the current scope. ``Bind`` has no body;
        the var is visible to all later statements in the same scope, which is
        also C's rule for a declaration in a block.
  * ``IfThenElse``       -> C ``if`` / ``else``
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

Dynamic shapes: any extent that is not a compile-time constant is emitted as a
trailing ``int <name>_dim`` parameter on the ``__KERNEL__`` and aliased in-body
as ``const int <name> = (int)<name>_dim;``, so every ``_render_index`` site works
unchanged. One compile serves every shape; the runtime resolves the values from
the input shapes at launch time. Constant extents stay folded as literals.
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
    shape: tuple[Any, ...]  # natural (row-major) buffer shape; entries may be
                            # int (static) or a TIR Var name (dynamic)
    is_output: bool


@dataclass
class ShapeDim:
    """One emitted ``dim4`` component: a literal or a runtime dim variable."""

    text: str            # C expression, e.g. "1024" or "M"


def _dim_text(s: Any) -> str:
    """Render one buffer-shape entry as a C expression."""
    try:
        return str(int(s))
    except (TypeError, ValueError):
        return str(s)


def shape_to_dim4(shape: tuple[Any, ...]) -> tuple[ShapeDim, ...]:
    """Pad a natural shape to 4D by prepending 1s (no permutation)."""
    seq = [_dim_text(s) for s in shape]
    while len(seq) < 4:
        seq.insert(0, "1")
    if len(seq) > 4:
        raise NotImplementedError(f"buffer rank >4 not supported: {seq}")
    return tuple(ShapeDim(text=t) for t in seq)


@dataclass
class PPLKernelInfo:
    """Everything the runtime needs to bind + launch a translated kernel."""

    kernel_name: str
    args: list[BufArg]
    source: str = ""
    # Dynamic (symbolic) buffer extents, in parameter-declaration order.  Each
    # gets its own trailing ``int`` argument on the emitted ``__KERNEL__``; the
    # runtime resolves their values from the input tensors' shapes at launch.
    dyn_dims: list[str] = field(default_factory=list)

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


def _is_symbolic(e: Any) -> bool:
    """True when a TIR extent is a symbolic var rather than a constant."""
    try:
        int(e)
        return False
    except (TypeError, ValueError):
        return True


def _is_zero(e: Any) -> bool:
    try:
        return int(e) == 0
    except (TypeError, ValueError):
        return False


def _num_stages(node: Any) -> int | None:
    """Return a ``For``'s ``num_stages`` annotation, or ``None`` if absent.

    ``T.Pipelined(n, num_stages=k)`` lowers to a plain serial ``For`` whose
    ``annotations`` carry ``num_stages``; a plain ``T.serial`` loop has no such
    tag.  This is the only signal distinguishing "please double/triple-buffer
    this loop" from "run it straight through", so it is what gates
    ``enable_pipeline()``.
    """
    ann = getattr(node, "annotations", None)
    if not ann:
        return None
    val = ann.get("num_stages")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _collect_binds(node: Any) -> list[Any]:
    """Collect every ``Bind`` statement in the kernel body (any nesting)."""
    out: list[Any] = []
    tn = type(node).__name__
    if tn == "Bind":
        out.append(node)
        return out
    if tn in ("For", "SBlock", "Block", "SBlockRealize", "BlockRealize"):
        return _collect_binds(node.body)
    if tn == "IfThenElse":
        out += _collect_binds(node.then_case)
        if getattr(node, "else_case", None) is not None:
            out += _collect_binds(node.else_case)
        return out
    if tn == "SeqStmt":
        for s in node.seq:
            out += _collect_binds(s)
    return out


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

    def __init__(self, func: Any, kernel_name: str, out_idx: list[int] | None = None):
        self.func = func
        self.kernel_name = kernel_name
        self.lines: list[str] = []
        self._tmp = 0

        # Grid dispatch state, populated by run(): the thread_binding axes, the
        # core axis (last of a 3-axis grid), the kernel's Bind statements, and
        # the recognized core partition (axis, per-core extent) if any.
        self.grid: list[Any] = []
        self.core_for: Any = None
        self._binds: list[Any] = []
        self.partition: tuple[Any, Any] | None = None

        # Classify buffers: globals (param buffer_map) vs locals (alloc_buffers).
        self.globals: dict[str, BufArg] = {}
        self.global_bufs: dict[str, Any] = {}
        self.locals: dict[str, tuple[tuple[int, int, int, int], str]] = {}

        # Which params are outputs.  The eager-trace path records this in the
        # `tilelang_out_idx` attr; the `@T.prim_func` path does not set it at
        # all, so the caller (tilelang.tpu.compiler.compile) passes it down.
        # A negative index counts from the end of the param list.
        if out_idx is None:
            out_idx = func.attrs.get("tilelang_out_idx", [-1])
        if isinstance(out_idx, int):
            out_idx = [out_idx]
        out_idx = [int(i) for i in out_idx]
        params = list(func.params)
        n = len(params)
        out_set = {i % n for i in out_idx}

        self.args: list[BufArg] = []
        self.dyn_names: list[str] = []  # symbolic extents, in declaration order
        for i, p in enumerate(params):
            buf = func.buffer_map[p]
            ppl_dt = _ppl_dtype(buf.dtype)
            shape = tuple(buf.shape)
            for s in shape:
                if _is_symbolic(s):
                    name = str(s)
                    if name not in self.dyn_names:
                        self.dyn_names.append(name)
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

    def _dim4_of(self, shape: tuple[Any, ...]) -> tuple[ShapeDim, ...]:
        """Natural shape -> dim4 entries (pad to 4D by prepending 1s).

        No ``ORDER`` permutation here: ``ORDER`` belongs to the *src* order of
        ``make_gtensor_permute`` (which the ``_mem_shape`` declaration feeds),
        not to the already-permuted logical shape.
        """
        return shape_to_dim4(shape)

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

    @staticmethod
    def _dim4_str_dyn(d4: tuple[ShapeDim, ...]) -> str:
        return "{" + ", ".join(sd.text for sd in d4) + "}"

    # -- runtime dim arguments --------------------------------------------- #

    def emit_dyn_dim_decls(self) -> None:
        """Materialize the runtime shape args as dim-variable aliases.

        The wrapper passes them as ``int`` (promoted from ``unsigned long long``);
        block-extent arithmetic keeps them in symbolically-rendered C
        expressions, so nothing else needs to know they are runtime values.
        """
        for name in self.dyn_names:
            self.emit(f"const int {name} = (int){name}_dim;")

    def emit_dyn_dim_comment(self) -> None:
        if self.dyn_names:
            self.emit(f"// runtime dims: {', '.join(self.dyn_names)}")

    # -- global tensor declarations ---------------------------------------- #

    def emit_global_decls(self) -> None:
        self.emit("int order[4] = {0, 2, 1, 3};")
        for arg in self.args:
            d4 = self._dim4_of(arg.shape)
            mem = ", ".join(sd.text for sd in shape_to_dim4(arg.shape))
            self.emit(f"dim4 {arg.name}_mem_shape = {{{mem}}};")
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
        elif tn == "Bind":
            # eager-builder `let` (e.g. bx_global = bc * M_tiles_per_core + bx).
            # Bind has no body; the var is visible to all later stmts in the same
            # scope, so mirror that with a C declaration at the current indent.
            #
            # When the enclosing grid axis was narrowed to this core's slice
            # (see _find_core_partition), the loop var already holds the global
            # index, so the slice-origin arithmetic is a no-op and must be
            # dropped -- keeping it would re-apply the offset and skip tiles.
            rewritten = self.partition_bind_rewrite(node)
            if rewritten is not None:
                self.emit(f"int {node.var.name} = {rewritten};")
            else:
                self.emit(f"int {node.var.name} = {_render_index(node.value)};")
        elif tn == "AttrStmt":
            # Overlap markers emitted by LowerTpuPipeline (tilelang/transform/
            # lower_tpu_pipeline.py).  The key *is* the call text, so this is
            # how ``parallel_start()`` / ``parallel_end()`` reach the output.
            key = str(node.attr_key)
            if key.startswith("ppl::") and key.endswith("()"):
                self.emit(f"{key[5:]};")
            else:
                self.emit_stmt(node.body)
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
            # ``T.Pipelined`` lowers to a plain serial For tagged with
            # num_stages; the tag itself is all we need, since PPL's
            # ``enable_pipeline()`` performs the multi-buffering and
            # load/compute overlap in the compiler.  It is a *hint* on the
            # loop, so it must be the first statement in the body -- and it
            # takes no depth argument, matching the PPL examples.
            if _num_stages(node) is not None:
                self.emit("enable_pipeline();")
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

    def partition_bind_rewrite(self, bind: Any) -> str | None:
        """Render ``bind`` for a narrowed axis, or ``None`` to render normally.

        For the recognized partition ``<bind var> = <core> * <per_core> + <axis>``
        the narrowed loop already yields the global index, so the bind is just
        the loop var. Any other expression in the same bind is left alone.
        """
        if self.partition is None:
            return None
        axis_for, _ = self.partition
        val = bind.value
        if type(val).__name__ != "Add":
            return None
        mul, axis_var = (val.a, val.b) if type(val.b).__name__ == "Var" else (val.b, val.a)
        if type(axis_var).__name__ != "Var" or axis_var.name != axis_for.loop_var.name:
            return None
        return axis_var.name

    def _find_core_partition(self) -> tuple[Any, Any] | None:
        """Recover a core-partitioned grid axis from the kernel's ``Bind`` set.

        A kernel that splits work across cores writes the slice origin as a
        ``let``, e.g. ``bx_global = bc * M_tiles_per_core + bx`` where ``bc`` is
        the core axis and ``M_tiles_per_core == ceildiv(M_tiles, core_num)``.
        That means grid axis ``bx`` is *meant* to run over one core's share, but
        the DSL records its extent as the whole tile count (``ceildiv(M,
        block_M)``) -- the per-core trim is left to a body guard.

        Emitting the grid axis at its full extent on every core therefore walks
        the entire tile space 4x and lets the guard cut a different amount per
        core, giving a 4:3:2:1 work staircase (even though every core is live).
        The reference emitter for this kernel instead bounds each core to
        ``tiles_per_core``, which is what makes the split even.

        Returns ``(axis_for, per_core_extent)`` when the pattern is recognized,
        else ``None``. Only a multiplier that *provably* equals
        ``ceildiv(axis_extent, core_num)`` is accepted, so an unrecognized
        partition is left exactly as written rather than silently rewritten.
        """
        if not self.core_for:
            return None
        core_name = self.core_for.loop_var.name
        core_ext = self.core_for.extent
        for bind in self._binds:
            val = bind.value
            if type(val).__name__ != "Add":
                continue
            mul, axis_var = (val.a, val.b) if type(val.b).__name__ == "Var" else (val.b, val.a)
            if type(mul).__name__ != "Mul" or type(axis_var).__name__ != "Var":
                continue
            core_side, mult = (mul.a, mul.b) if type(mul.a).__name__ == "Var" else (mul.b, mul.a)
            if type(core_side).__name__ != "Var" or core_side.name != core_name:
                continue
            axis = next((g for g in self.grid if g.loop_var.name == axis_var.name), None)
            if axis is None:
                continue
            # Check multiplier == ceildiv(axis_extent, core_num) *by value*, not
            # by text: the DSL's flooring shape (a-1)//b differs from the naive
            # ceildiv form, so compare as expressions via the simplifier.
            try:
                want = _tir.ceildiv(axis.extent, core_ext)
                if _tir.analysis.expr_deep_equal(want, mult):
                    return axis, mult
            except Exception:
                continue
            # The eager-builder folds ceildiv into FloorDiv-by-c; accept that
            # exact shape too (extent + core - 1) // core.
            tn = type(mult).__name__
            if tn == "FloorDiv" and _render_index(mult.b) == _render_index(core_ext):
                try:
                    if _tir.analysis.expr_deep_equal(mult.a, axis.extent + core_ext - 1):
                        return axis, mult
                except Exception:
                    continue
        return None

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
        self.grid = list(grid)
        self.core_for = grid[-1] if len(grid) >= 3 else None
        self._binds = _collect_binds(body)

        # kernel signature: pointer args, then one int per symbolic extent.
        # Dyn dims carry no C default: PPL binds this entry point by name and
        # always passes them explicitly (see ppl_runner._generic_wrapper_src).
        sig_ptrs = ", ".join(f"{a.ppl_dtype} *{a.ptr_name}" for a in self.args)
        sig_dims = ", ".join(f"int {n}_dim" for n in self.dyn_names)
        sig = ", ".join(s for s in (sig_ptrs, sig_dims) if s)
        self.lines.append('#include "ppl.h"')
        self.lines.append('#include "ppl_wrapper_func.h"')
        self.lines.append("")
        self.lines.append("using namespace ppl;")
        self.lines.append("")
        self.lines.append(f"__KERNEL__ void {self.kernel_name}({sig}) {{")

        self.emit_dyn_dim_decls()
        self.emit_global_decls()

        # Grid dispatch. SG2260E has 4 cores and PPL is SPMD: the kernel body
        # runs once per core and each core must claim a slice. Emitting the grid
        # axes as plain C loops instead runs the whole grid on *one* core, which
        # is correct but 4x slower -- the profiler shows the other three cores
        # idle. So a 3-axis grid (blockIdx.z == core count) is lowered to a
        # get_block_index() slice rather than a loop.
        multicore = self.core_for is not None
        indent_base = 2
        if multicore:
            core_for = self.core_for
            core_var = core_for.loop_var.name
            core_ext = _render_index(core_for.extent)
            self.emit("set_block_num_max();", indent_base)
            self.emit(f"const int {core_var} = get_block_index();", indent_base)
            self.emit(f"if ({core_var} >= {core_ext}) return;", indent_base)
            self.emit("{", indent_base)
            indent_base += 1
            grid = grid[:-1]

        # A core-partitioned axis must be bounded to this core's share. If we
        # emit it at its full extent instead, every core walks the whole tile
        # space and only the body's guard trims the tail -- which cuts a
        # different amount per core and yields a 4:3:2:1 staircase. The
        # partition is only applied where it is provably what the kernel asked
        # for (see _find_core_partition); otherwise the axis is emitted as-is.
        partition = self._find_core_partition() if multicore else None
        self.partition = partition
        for g in grid:
            var = g.loop_var.name
            ext = _render_index(g.extent)
            if partition is not None and g is partition[0]:
                per_core = _render_index(partition[1])
                self.emit(
                    f"int {var}_start = {core_var} * ({per_core});", indent_base
                )
                self.emit(
                    f"int {var}_end = min({var}_start + ({per_core}), ({ext}));",
                    indent_base,
                )
                self.emit(
                    f"for (int {var} = {var}_start; {var} < {var}_end; {var}++) {{",
                    indent_base,
                )
            else:
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
        if multicore:
            indent_base -= 1
            self.emit("}", indent_base)
        self.lines.append("}")

        # __TEST__ stub
        self.lines.append("")
        self._emit_test_stub()

        source = "\n".join(self.lines) + "\n"
        return PPLKernelInfo(
            kernel_name=self.kernel_name,
            args=self.args,
            source=source,
            dyn_dims=list(self.dyn_names),
        )

    _TEST_STUB_DIM = 1024

    def _emit_test_stub(self) -> None:
        """Emit the standalone ``__TEST__`` launcher.

        Symbolic extents have no value to fall back on, so the stub picks a
        concrete one per dim (``_TEST_STUB_DIM``, tile-aligned for the shapes
        used here) purely so ``--gen_test`` has a well-typed call to emit.  The
        Python runtime never invokes this path -- it drives ``__KERNEL__``
        directly with the real dims.
        """
        self.lines.append(f"__TEST__ void {self.kernel_name}_main() {{")
        for n in self.dyn_names:
            self.lines.append(f"  int {n} = {self._TEST_STUB_DIM};")
        call_args = []
        for a in self.args:
            mem = ", ".join(sd.text for sd in shape_to_dim4(a.shape))
            self.lines.append(f"  dim4 {a.name}_ts = {{{mem}}};")
            self.lines.append(
                f"  {a.ppl_dtype} *{a.ptr_name} = ppl::malloc<{a.ppl_dtype}>(&{a.name}_ts);")
            if not a.is_output:
                self.lines.append(f"  ppl::rand({a.ptr_name}, &{a.name}_ts, -1.0, 1.0);")
            call_args.append(a.ptr_name)
        call_args.extend(f"{n}" for n in self.dyn_names)
        self.lines.append(f"  {self.kernel_name}({', '.join(call_args)});")
        self.lines.append("}")


# Names the PPL/C front end treats specially.  `main` is the sharp one: a
# `@T.prim_func` nested inside a jit factory typically inherits `global_symbol
# == "main"` from the inner Python function's name, and emitting `__KERNEL__
# void main(...)` makes the C front end reject the file with "too many
# parameters (N) for 'main': must be 0, 2, or 3".
_RESERVED_KERNEL_NAMES = {
    "main", "printf", "malloc", "free", "exit", "abort", "assert",
    "memcpy", "memset", "rand", "srand", "sqrt", "exp", "log", "pow",
    "min", "max", "abs", "floor", "ceil", "round",
}


def _sanitize_kernel_name(name: str) -> str:
    """Return a name safe to emit as a PPL ``__KERNEL__`` entry point."""
    clean = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(name))
    if not clean or clean[0].isdigit():
        clean = "tl_" + clean
    if clean in _RESERVED_KERNEL_NAMES:
        clean += "_kernel"
    return clean


def translate(
    func: Any,
    kernel_name: str | None = None,
    out_idx: int | list[int] | None = None,
) -> PPLKernelInfo:
    """Translate a lowered tilelang PrimFunc into a PPL kernel + arg metadata.

    *out_idx* names the output param(s) (negative counts from the end).  When
    omitted the ``tilelang_out_idx`` attr is used if present, else the last
    param.
    """
    if kernel_name is None:
        kernel_name = str(func.attrs.get("global_symbol", "tl_kernel"))
    return PPLEmitter(func, _sanitize_kernel_name(kernel_name), out_idx=out_idx).run()
