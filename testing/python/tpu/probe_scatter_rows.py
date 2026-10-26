"""Probe: ``tl_tpu_scatter_rows`` writes tile rows into a global buffer by index.

Isolates the one new codegen primitive the fused qkv/rope kernel depends on
(``ppl_codegen.emit_scatter_rows``) so a failure there is not confounded with
the gemm / norm / rope arithmetic.  The kernel is deliberately shaped like its
use site: a grid of *jobs*, each owning ``rows`` contiguous source rows and a
``[1, rows]`` uint32 index table, scattering two tiles per job back to back
(the second scatter is what would catch L2-scratch reuse races).

Contract under test -- all four are asserted, and each fails differently:

1. **Addressed rows land.**  ``dst[idx[t]] == src[t]`` for every scattered row.
2. **Untouched rows are untouched**, bit for bit.  A mistaken row pitch, a
   wrong destination base (e.g. a permuted gtensor instead of a flat one), or a
   spill past the tile end all show up here.
3. **A repeat run reproduces run 1 exactly.**  The L2 staging slices are reused
   every iteration; without the trailing ``sync_engine`` the next ``dma::store``
   races the in-flight scatter and the second run differs.
4. **Duplicate indices do not hang.**  The fused kernel's padding rows all map
   to one reserved slot, so the hardware must tolerate a repeated index.  The
   winner is unspecified; the row must simply equal one of the candidates.

The destination extent is *dynamic* and appears only in an output shape, which
is how the fused kernel declares its cache (``S2``).  Passing the destination
preallocated is what resolves it, so this probe also covers that launch path.

Run on real hardware under the PPL env (see the ``ppl-tpu-run-recipe`` memo):

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/probe_scatter_rows.py --run

Without ``--run`` it only emits + prints the kernel source (compile smoke test).
"""

import os
import sys

import tilelang
import tilelang.language as T

#: Flat row count of the destination -- output-only, like the paged cache.
S2 = T.dynamic("S2", "int32")

_NUM_CORES = 4


def _logical_device() -> int:
    """The torch *logical* device id to run on (see ``test_flash_attention_paged``)."""
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


@tilelang.jit(target="tpu")
def scatter_rows_probe(
    rows: int = 64,
    width: int = 64,
    n_job: int = 8,
    dtype: T.dtype = T.bfloat16,
):
    """Scatter ``n_job`` tiles of ``[rows, width]`` into ``Dst [S2, width]``."""
    assert n_job % _NUM_CORES == 0, "one equal slice of jobs per core"
    n_rows = n_job * rows
    groups_per_core = n_job // _NUM_CORES

    @T.prim_func
    def main(
        SrcA: T.Tensor((n_rows, width), dtype),
        IdxA: T.Tensor((1, n_rows), T.uint32),
        SrcB: T.Tensor((n_rows, width), dtype),
        IdxB: T.Tensor((1, n_rows), T.uint32),
        Dst: T.Tensor((S2, width), dtype),
    ):
        with T.Kernel(n_job, 1, _NUM_CORES) as (bx, by, bc):
            a = T.alloc_local((rows, width), dtype)
            b = T.alloc_local((rows, width), dtype)
            ia = T.alloc_local((1, rows), T.uint32)
            ib = T.alloc_local((1, rows), T.uint32)

            g = bc * groups_per_core + bx
            with T.If(g < n_job), T.Then():
                row0 = g * rows
                T.copy(SrcA[row0, 0], a)
                T.copy(IdxA[0, row0], ia)
                T.call_extern("handle", "tl_tpu_scatter_rows", Dst.data, ia.data, a.data)
                # Second scatter, same core, same L2 slices: this is the pair
                # that exposes a missing engine fence.
                T.copy(SrcB[row0, 0], b)
                T.copy(IdxB[0, row0], ib)
                T.call_extern("handle", "tl_tpu_scatter_rows", Dst.data, ib.data, b.data)

    return main


def _check(name: str, ok: bool) -> bool:
    print(f"  {name:<44}{'ok' if ok else 'FAIL'}")
    return ok


def _run(kernel, rows, width, n_job, do_source):
    import torch

    dev = f"tpu:{_logical_device()}"
    torch.manual_seed(0)
    n_rows = n_job * rows
    s2 = 8 * n_rows  # roomy: the untouched-row check needs spare rows

    src_a = torch.randn(n_rows, width, dtype=torch.bfloat16)
    src_b = torch.randn(n_rows, width, dtype=torch.bfloat16)

    def _table(order):
        return order.view(1, -1).contiguous()

    # Disjoint destinations for A and B, so neither can mask the other.  Rows
    # are drawn from a permutation that skips ``dup_target`` (used below), so the
    # duplicate-index case cannot be contaminated by an ordinary write.
    dup_target = 3
    perm = torch.randperm(s2 - 1)[: 2 * n_rows]
    perm = torch.where(perm >= dup_target, perm + 1, perm)
    idx_a, idx_b = perm[:n_rows], perm[n_rows:]

    dst0 = torch.randn(s2, width, dtype=torch.bfloat16)

    a_dev, b_dev = src_a.to(dev), src_b.to(dev)
    ia_dev = _table(idx_a).to(dev)
    ib_dev = _table(idx_b).to(dev)
    dst_dev = dst0.to(dev)

    kernel(a_dev, ia_dev, b_dev, ib_dev, dst_dev)  # preallocated output
    run1 = dst_dev.cpu()
    dst_dev2 = dst0.to(dev)
    kernel(a_dev, ia_dev, b_dev, ib_dev, dst_dev2)
    run2 = dst_dev2.cpu()

    if do_source:
        print(kernel.get_kernel_source())

    expect = dst0.clone()
    expect[idx_a] = src_a
    expect[idx_b] = src_b

    touched = torch.zeros(s2, dtype=torch.bool)
    touched[idx_a] = True
    touched[idx_b] = True

    ok = True
    ok &= _check(
        "addressed rows match",
        torch.equal(run1[idx_a], src_a) and torch.equal(run1[idx_b], src_b),
    )
    ok &= _check(
        "untouched rows byte-identical",
        torch.equal(run1[~touched], dst0[~touched]),
    )
    ok &= _check("repeat run identical", torch.equal(run1, run2))
    ok &= _check("whole buffer matches reference", torch.equal(run1, expect))
    if not torch.equal(run1, expect):
        bad = (run1 != expect).any(dim=1).nonzero().flatten()
        t_a = int((run1[idx_a] != src_a).any(dim=1).sum())
        t_b = int((run1[idx_b] != src_b).any(dim=1).sum())
        print(f"    mismatched rows: {bad[:12].tolist()} ({bad.numel()} total)")
        print(f"    of which addressed: A {t_a}/{n_rows}  B {t_b}/{n_rows}")

    # --- duplicated indices (padded rows share the reserve slot) ----------- #
    # Every job's first source row points at one destination row; the kernel
    # must complete and write one of those candidates.
    idx_a_dup = torch.zeros_like(idx_a)
    idx_a_dup[:n_job] = dup_target  # at least one source row per job
    idx_a_dup[n_job:] = idx_a[: n_rows - n_job]
    dst_dup = dst0.clone().to(dev)
    kernel(a_dev, _table(idx_a_dup).to(dev), b_dev, ib_dev, dst_dup)
    got = dst_dup.cpu()[dup_target]
    cands = src_a[:n_job]
    ok &= _check(
        "duplicate index does not hang; winner is a source row",
        any(torch.equal(got, c) for c in cands),
    )

    return ok


def main():
    do_run = "--run" in sys.argv
    do_source = "--source" in sys.argv
    rows, width, n_job = 64, 64, 8

    kernel = scatter_rows_probe(rows=rows, width=width, n_job=n_job)
    if not do_run:
        print(kernel.get_kernel_source() if do_source else "compiled (--run to execute)")
        return 0

    import torch
    import torch_tpu  # noqa: F401  (must precede the first device tensor)

    torch.tpu.set_device(_logical_device())
    print(f"logical device {torch.tpu.current_device()}")

    ok = _run(kernel, rows, width, n_job, do_source)
    print("PASS: scatter_rows matches reference" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
