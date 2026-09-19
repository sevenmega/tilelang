"""Real software pipelining for TPU ``T.Pipelined`` loops.

``T.Pipelined(n, num_stages=k)`` lowers to a plain serial ``For`` carrying
``annotations={"num_stages": k}``.  On the TPU path the emitter previously
translated that tag into PPL's ``enable_pipeline()`` hint and let the PPL
compiler choose the schedule — which turned out to yield a *one*-deep
schedule regardless of ``k`` (``enable_pipeline()`` takes no depth argument).

This pass builds the pipeline explicitly instead, so the buffer *count* is
owned by the tilelang compiler.  For a loop of trip count ``T`` tagged
``num_stages = S``:

* statements writing a local buffer that another statement in the same loop
  reads are the **prefetch** stage (the ``T.copy`` global->local loads);
* every other statement is the **compute** stage (the ``T.gemm``);
* each buffer written by the prefetch stage is cloned ``S`` times
  (``A_local_v0 .. A_local_v{S-1}``) — a rotating set of tiles;
* the loop is rebuilt as prologue / steady / drain:

  - prologue: load tiles ``0 .. S-2`` with a per-tile guard
  - steady:   ``T // S`` iterations, unrolled exactly ``S`` times.  Step
    ``i*S + j`` loads tile ``i*S + j + S`` and computes tile ``i*S + j``,
    both into version ``j``.  A steady iteration consumes exactly ``S``
    tiles and ``S`` versions exist, so the version offset is the
    *compile-time constant* ``j`` — that is what lets a loop with a dynamic
    trip count still name every buffer statically (PPL rejects runtime
    buffer selection outright).
  - drain:    the trailing ``T % S`` tiles, computed out of the versions the
    steady loop left behind, one guard per step.

Each load/compute pair is bracketed by ``parallel_start()`` /
``parallel_end()``, matching the hand-written double-buffer template
(``_PL_TEMPLATE_MULTIBUF``) so the backend overlaps the DMA with the matmul.

The pass is invoked explicitly from ``tilelang/tpu/compiler.py`` right before
``emit_pl`` — the TPU path bypasses the generic pass pipeline, so registering
it there would leave it dead.
"""

from __future__ import annotations

from typing import Any

from tvm.tirx import (
    AttrStmt,
    BufferLoad,
    BufferStore,
    Evaluate,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    PrimFunc,
    SeqStmt,
    SBlock,
    SBlockRealize,
    StringImm,
    Var,
    decl_buffer,
)
from tvm.tirx.stmt_functor import ir_transform, post_order_visit
from tvm.tirx.transform import prim_func_pass

# Marker keys recognised by the PPL emitter (tilelang/tpu/ppl_codegen.py).
MARK_PARALLEL_START = "ppl::parallel_start()"
MARK_PARALLEL_END = "ppl::parallel_end()"


# --------------------------------------------------------------------------- #
# small helpers                                                                #
# --------------------------------------------------------------------------- #


def _seq(stmts: list[Any]) -> Any:
    """Flatten statements into one valid TIR statement.

    ``SeqStmt`` refuses 0- and 1-element lists, so a single statement is
    returned bare and callers must not pass an empty list.
    """
    stmts = [s for s in stmts if s is not None]
    if not stmts:
        raise ValueError("_seq() requires at least one statement")
    if len(stmts) == 1:
        return stmts[0]
    return SeqStmt(stmts)


def _marker(key: str) -> AttrStmt:
    """An ``AttrStmt`` the emitter renders as a bare PPL call."""
    return AttrStmt(Var("ppl_marker", "handle"), key, StringImm(key), Evaluate(IntImm("int32", 0)))


def _num_stages(loop: For) -> int | None:
    ann = getattr(loop, "annotations", None)
    if not ann:
        return None
    val = ann.get("num_stages")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _stmts_of(body: Any) -> list[Any]:
    return list(body.seq) if type(body).__name__ == "SeqStmt" else [body]


def _children(node: Any) -> list[Any]:
    """Direct TIR children of a statement or expression."""
    tn = type(node).__name__
    if tn in ("SeqStmt",):
        return list(node.seq)
    if tn in ("For", "SBlock", "Block"):
        return [node.body]
    if tn in ("SBlockRealize", "BlockRealize"):
        return [node.block]
    if tn == "IfThenElse":
        out = [node.then_case]
        if getattr(node, "else_case", None) is not None:
            out.append(node.else_case)
        return out
    if tn == "AttrStmt":
        return [node.body]
    if tn in ("Evaluate",):
        return [node.value]
    if tn in ("Call", "Add", "Sub", "Mul", "Div", "Mod", "FloorDiv", "FloorMod",
              "Min", "Max", "EQ", "NE", "LT", "LE", "GT", "GE", "And", "Or"):
        return list(getattr(node, "args", [])) or [node.a, node.b]
    if tn == "BufferStore":
        return [node.value, *node.indices]
    if tn == "BufferLoad":
        return list(node.indices)
    if tn == "BufferRegion":
        return []
    return []


# Per-tileop region read/write roles, as consumed by the emitter
# (tilelang/tpu/ppl_codegen.py): an argument listed here is read or written
# wholesale at the region level.  Tile ops encode their accesses in
# ``tl.tileop.region`` calls rather than in BufferStore nodes, so a plain
# BufferStore walk sees nothing.
_TILEOP_ROLES: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "tl.tileop.copy": ((0,), (1,)),
    "tl.tileop.fill": ((), (0,)),
    "tl.tileop.gemm": ((0, 1), (2,)),
    "tl.tileop.reduce": ((0,), (1,)),
}


def _region_calls(stmt: Any) -> list[Any]:
    out: list[Any] = []

    def visit(node: Any) -> None:
        if type(node).__name__ == "Call" and getattr(node.op, "name", "") == "tl.tileop.region":
            out.append(node)

    post_order_visit(stmt, visit)
    return out


def _tileop_accesses(stmt: Any, local_names: set[str]) -> tuple[set[str], set[str]]:
    """(reads, writes) of local buffers by one statement."""
    reads: set[str] = set()
    writes: set[str] = set()

    def note(call: Any, roles: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
        read_slots, write_slots = roles
        for slot in read_slots:
            if slot < len(call.args) and type(call.args[slot]).__name__ == "Call":
                buf = call.args[slot].args[0].buffer
                if buf.name in local_names:
                    reads.add(buf.name)
        for slot in write_slots:
            if slot < len(call.args) and type(call.args[slot]).__name__ == "Call":
                buf = call.args[slot].args[0].buffer
                if buf.name in local_names:
                    writes.add(buf.name)

    def visit(node: Any) -> None:
        if type(node).__name__ != "Call":
            return
        name = getattr(node.op, "name", "")
        if name == "tl.tileop.region":
            # The region calls *inside* a tile op; only the enclosing tile op
            # knows whether its region is read or written.
            return
        roles = _TILEOP_ROLES.get(name)
        if roles is not None:
            note(node, roles)

    post_order_visit(stmt, visit)

    # Plain stores (the elementwise ReLU loop, scalar writes) still count, but
    # the BufferLoad inside a ``tl.region`` is a *handle* to the buffer, not a
    # read — its direction comes from the enclosing tile op — so the walk does
    # not descend into region calls.
    def scan(node: Any, fn: Any) -> None:
        if type(node).__name__ == "Call" and getattr(node.op, "name", "") == "tl.tileop.region":
            return
        fn(node)
        for child in _children(node):
            scan(child, fn)

    def is_store(node: Any) -> None:
        if type(node).__name__ == "BufferStore" and node.buffer.name in local_names:
            writes.add(node.buffer.name)

    def is_load(node: Any) -> None:
        if type(node).__name__ == "BufferLoad" and node.buffer.name in local_names:
            reads.add(node.buffer.name)

    scan(stmt, is_store)
    scan(stmt, is_load)
    return reads, writes


def _split_stages(loop: For, local_names: set[str]) -> tuple[list[Any], list[Any], set[str]]:
    """Split a loop body into (prefetch, compute, multi-buffered locals).

    A local written by one statement and read by *another* statement in the
    same loop is a cross-statement producer — that is what needs
    multi-buffering.  ``C_local`` is only ever touched by the gemm itself
    (accumulated in place), so it stays a single register; ``A_local`` /
    ``B_local`` are loaded by ``T.copy`` and consumed by ``T.gemm``, so they
    are versioned.
    """
    stmts = _stmts_of(loop.body)
    access = [_tileop_accesses(s, local_names) for s in stmts]

    prefetch: list[Any] = []
    compute: list[Any] = []
    versioned: set[str] = set()
    for i, stmt in enumerate(stmts):
        _reads_i, writes_i = access[i]
        produced = False
        for name in writes_i:
            # Read by some *other* statement in this loop -> this statement
            # feeds the next one, so it is the pipeline's prefetch stage.
            if any(name in access[k][0] for k in range(len(stmts)) if k != i):
                produced = True
                break
        if produced:
            prefetch.append(stmt)
            versioned |= writes_i
        else:
            compute.append(stmt)
    return prefetch, compute, versioned


def _rebind(stmt: Any, versions: dict[str, Any], var: Var, value: Any) -> Any:
    """Point ``stmt`` at versioned clones and substitute ``var -> value``."""

    def rewrite(node: Any) -> Any:
        tn = type(node).__name__
        if tn == "BufferStore":
            return BufferStore(versions.get(node.buffer.name, node.buffer), node.value, node.indices)
        if tn == "BufferLoad":
            return BufferLoad(versions.get(node.buffer.name, node.buffer), node.indices)
        return None

    versioned_stmt = ir_transform(stmt, lambda n: None, rewrite)
    return _subst_var(versioned_stmt, var, value)


def _guard_loads(stmt: Any, tile: Any, trip: Any) -> Any:
    """Wrap a prefetch statement in ``if tile < trip``.

    The steady loop prefetches S-1 tiles ahead of what it computes, so on the
    last iteration it asks for tiles that do not exist whenever the trip count
    is not a multiple of S.  The DMA would then read past the end of A/B, so
    the load is predicated on its tile being in range.
    """
    return IfThenElse(tile < trip, stmt, None)


def _subst_var(stmt: Any, var: Var, value: Any) -> Any:
    """Replace ``var`` with ``value`` inside ``stmt``.

    Implemented directly rather than via ``stmt_functor.substitute``: the
    latter raises on some region-encoded arguments, while a visit-and-rebuild
    over ``Var`` nodes reaches every occurrence (including the ones nested in
    ``tl.tileop.region`` calls, which is where tile offsets live).
    """

    def rewrite(node: Any) -> Any:
        if type(node).__name__ == "Var" and node.name == var.name:
            return value
        return None

    return ir_transform(stmt, lambda n: None, rewrite)


def _clone_block(blk: Any, body: Any, allocs: list[Any]) -> Any:
    """Copy an ``SBlock`` with a new body / alloc list, field names permitting."""
    kwargs: dict[str, Any] = {}
    for field in ("init", "match_buffers", "annotations"):
        if hasattr(blk, field):
            kwargs[field] = getattr(blk, field)
    return SBlock(
        blk.iter_vars,
        blk.reads,
        blk.writes,
        blk.name_hint,
        body,
        kwargs.get("init"),
        allocs,
        kwargs.get("match_buffers"),
        kwargs.get("annotations", {}),
        getattr(blk, "span", None),
    )


# --------------------------------------------------------------------------- #
# the transform                                                                #
# --------------------------------------------------------------------------- #


class _PipelineRewriter:
    def __init__(self, func: PrimFunc):
        self.func = func
        self.clones: list[Any] = []
        self.versioned_names: set[str] = set()

    # -- entry point ------------------------------------------------------- #

    def run(self) -> PrimFunc:
        loop = self._find_loop(self.func.body)
        if loop is None:
            return self.func

        local_names = self._local_names()
        prefetch, compute, versioned = _split_stages(loop, local_names)
        if not prefetch or not compute or not versioned:
            # Nothing cross-statement to overlap; leave the loop alone so the
            # emitter still applies its ``enable_pipeline()`` hint.
            return self.func

        stages = _num_stages(loop)
        assert stages is not None
        self.versioned_names = versioned

        for name in sorted(versioned):
            orig = self._buffer_by_name(name)
            if orig is None:
                return self.func
            for v in range(stages):
                self.clones.append(
                    decl_buffer(orig.shape, orig.dtype, name=f"{name}_v{v}", scope=orig.scope())
                )

        replacement = self._build_loop(loop, prefetch, compute, versioned, stages)
        if replacement is None:
            return self.func

        new_body = self._rebuild(self.func.body, loop, replacement)
        return self.func.with_body(new_body)

    @staticmethod
    def _find_loop(node: Any) -> For | None:
        found: list[For] = []

        def visit(n: Any) -> None:
            if type(n).__name__ == "For":
                s = _num_stages(n)
                if s is not None and s >= 2:
                    found.append(n)

        post_order_visit(node, visit)
        return found[0] if found else None

    # -- loop construction ------------------------------------------------- #

    def _build_loop(
        self,
        loop: For,
        prefetch: list[Any],
        compute: list[Any],
        versioned: set[str],
        stages: int,
    ) -> Any:
        s = stages
        step = IntImm("int32", s)
        trip = loop.extent  # ceildiv(K, block_K) — may be a dynamic expression
        kfull = trip // step
        prologue = self._prologue(prefetch, versioned, trip, loop, s)
        steady = self._steady(prefetch, compute, versioned, kfull, step, loop, s)
        drain = self._drain(compute, versioned, kfull, step, loop, s)
        return _seq([p for p in (prologue, steady, drain) if p is not None])

    def _version_map(self, versioned: set[str], v: int) -> dict[str, Any]:
        # clones were appended in sorted-name order, S per name
        out: dict[str, Any] = {}
        for name in sorted(versioned):
            out[name] = self._clone_of(name, v)
        return out

    def _clone_of(self, name: str, v: int) -> Any:
        for c in self.clones:
            if c.name == f"{name}_v{v}":
                return c
        raise KeyError(name)

    # -- prologue: tiles 0 .. S-2 ------------------------------------------ #

    def _prologue(
        self, prefetch: list[Any], versioned: set[str], trip: Any, loop: For, s: int
    ) -> Any:
        # The tile index is a literal here, so rewriting the loop variable = the
        # constant is what makes the offsets fold to compile-time values.
        var = loop.loop_var
        body: list[Any] = []
        for j in range(s - 1):
            tile = IntImm("int32", j)
            loads = [_rebind(st, self._version_map(versioned, j), var, tile) for st in prefetch]
            body.append(IfThenElse(tile < trip, _seq(loads), None))
        return _seq(body)

    # -- steady: T // S steps, unrolled S times ---------------------------- #

    def _steady(
        self,
        prefetch: list[Any],
        compute: list[Any],
        versioned: set[str],
        kfull: Any,
        step: Any,
        loop: For,
        s: int,
    ) -> Any:
        var = loop.loop_var
        body: list[Any] = []
        for j in range(s):
            # `here` is the tile this substep computes; the load targets the
            # tile that will be computed in the *next* substep, i.e. S-1 steps
            # ahead of `here` in global-step terms.
            here = var * step + IntImm("int32", j)
            ahead = here + IntImm("int32", s - 1)
            # Slot of the compute in this substep vs. slot the load writes.
            # They must differ: both sit inside one parallel_start/end block,
            # so a load into the slot being read would clobber the operand out
            # from under the MMA.  The slot that is safe to reuse is the one
            # the *previous* substep read, which is (j-1) mod S.
            load_slot = (j - 1) % s
            lmap = self._version_map(versioned, load_slot)
            cmap = self._version_map(versioned, j)
            loads = [_rebind(st, lmap, var, ahead) for st in prefetch]
            # The last steady iteration wants to prefetch tile `step + S - 1`,
            # which does not exist when the trip count is not a multiple of S;
            # clamp it so the DMA never runs off the end of A/B.
            guarded = [_guard_loads(st, ahead, loop.extent) for st in loads]
            pair: list[Any] = [
                _marker(MARK_PARALLEL_START),
                *guarded,
                *[_rebind(st, cmap, var, here) for st in compute],
                _marker(MARK_PARALLEL_END),
            ]
            body.append(_seq(pair))
        return For(var, IntImm("int32", 0), kfull, ForKind.SERIAL, _seq(body))

    # -- drain: the trailing T % S tiles ----------------------------------- #

    def _drain(
        self,
        compute: list[Any],
        versioned: set[str],
        kfull: Any,
        step: Any,
        loop: For,
        s: int,
    ) -> Any:
        var = loop.loop_var
        body: list[Any] = []
        for j in range(s):
            # Global step index of this trailing tile: Ki*S + j, where Ki is
            # the steady trip count.  The slot follows the same rule as the
            # steady loop -- tile `step` lives in slot `step % S`, which is
            # just `j` here because Ki*S is a multiple of S.
            here = kfull * step + IntImm("int32", j)
            vmap = self._version_map(versioned, j)
            comps = [_rebind(st, vmap, var, here) for st in compute]
            pair = [_marker(MARK_PARALLEL_START), *comps, _marker(MARK_PARALLEL_END)]
            body.append(IfThenElse(here < loop.extent, _seq(pair), None))
        return _seq(body)

    # -- buffer bookkeeping ------------------------------------------------ #

    def _local_names(self) -> set[str]:
        out: set[str] = set()
        for blk in self._blocks(self.func.body):
            for b in blk.alloc_buffers or []:
                out.add(b.name)
        return out

    def _buffer_by_name(self, name: str) -> Any:
        for blk in self._blocks(self.func.body):
            for b in blk.alloc_buffers or []:
                if b.name == name:
                    return b
        return None

    @staticmethod
    def _blocks(node: Any) -> list[Any]:
        out: list[Any] = []

        def visit(n: Any) -> None:
            if type(n).__name__ in ("SBlock", "Block"):
                out.append(n)

        post_order_visit(node, visit)
        return out

    # -- splice the new loop in and declare the clones --------------------- #

    def _rebuild(self, node: Any, old: Any, new: Any) -> Any:
        """Replace the pipelined loop with `new` and declare the clones.

        Matching is by *shape*, not by Python identity: the C++ mutator
        rebuilds every node it descends into, so the ``For`` handed to the
        post-order callback is a fresh object and ``is`` would never fire.
        ``_num_stages`` is the same predicate ``_find_loop`` used, so exactly
        the one loop we rewrote matches.
        """
        clones = list(self.clones)
        spliced: list[bool] = []

        def swap(s: Any) -> Any:
            """Replace the pipelined loop; None means 'unchanged' here."""
            if type(s).__name__ == "For" and _num_stages(s) is not None:
                spliced.append(True)
                return new
            return None

        def post(n: Any) -> Any:
            # Only the SBlock branch may fire.  ``ir_transform`` visits a block
            # as well as the SBlockRealize wrapping it, so handling both would
            # rebuild -- and therefore re-clone -- the *same* block twice, which
            # shows up as duplicate ``{name}_v{v}_bs`` / ``{name}_v{v}``
            # declarations in the emitted .pl and aborts ppl-compile.
            if type(n).__name__ != "SBlock":
                return None
            # The lowered kernel nests two blocks (``root`` wrapping
            # ``tilelang_root``); the loop lives in the inner one.  ``swap``
            # fires exactly once -- in the block that actually holds the loop --
            # so that, not block identity, is what decides whether to rebuild
            # and where to declare the clones.  (A subtree probe would be
            # useless here: by the time this callback runs the loop is already
            # gone from the rebuilt body.)
            spliced.clear()
            body = ir_transform(n.body, lambda c: None, swap)
            if not spliced:
                return None
            return _clone_block(n, body, list(n.alloc_buffers or []) + clones)

        return ir_transform(node, lambda n: None, post)


def LowerTpuPipeline() -> Any:
    """Real ``num_stages`` pipelining for TPU ``T.Pipelined`` tile loops."""

    def pass_fn(func: PrimFunc, mod, ctx):
        return _PipelineRewriter(func).run()

    return prim_func_pass(pass_fn, opt_level=0)


def lower_tpu_pipeline(func: PrimFunc) -> PrimFunc:
    """``LowerTpuPipeline`` applied straight to a PrimFunc.

    ``PrimFuncPass`` only runs through an ``IRModule``; the TPU compile path
    has a bare PrimFunc in hand, so it goes through this helper.
    """
    return _PipelineRewriter(func).run()
