"""Standalone paged/varlen flash-attention TPU kernel test.

Compiles the **real tilelang DSL kernel** ``flash_attention_paged`` (see
:mod:`tilelang.tpu.kernels.attention_paged`) through the generic TIR->PPL
translator and checks it against a plain torch reference from *gathered* K/V rows.

Unlike ``test_flash_attention.py`` (which is handed a dense padded
``[B, Hkv, Skv, D]`` K/V batch), this kernel reads the KV cache in place:

    Q     [T, Hq, D]        packed queries of the whole batch, contiguous
    Kc/Vc [Slots, Hkv, D]   the NHD paged cache exactly as stored
    Idx   [B, Hkv, KV]      flat rows of ``[Slots * Hkv, D]``: ``slot * Hkv + h``
    Mask  [B, QP, KV]       fp32 block-diagonal additive mask
    QOff  [B]               first query row of each request in Q
    Out   [B, QP, Hq, D]    per-request padded result

so one launch covers requests of *different* lengths and there is no driver-side
gather, pack or scatter.  ``QP`` / ``KV`` are bucket-wide (every request shares
them) and must be multiples of ``block_M`` / ``block_N``.

Run on real hardware under the PPL env (see the ``ppl-tpu-run-recipe`` memo):

    TPU_VISIBLE_DEVICES=<chip> python testing/python/tpu/test_flash_attention_paged.py --run

Without ``--run`` it only emits + prints the kernel source (compile smoke test).
Add ``--profile`` to profile the launch (implies ``--run``); ``--verbose``
forwards to ``bigTpuProfile``.
"""

import os
import sys

from tilelang.tpu.kernels.attention_paged import flash_attention_paged

MASK_NEG = -30000.0  # fp16-safe stand-in for -inf


def _logical_device() -> int:
    """The torch *logical* device id to run on.

    ``ensure_tpu_device()`` alone is not enough on a machine where chip 0 is
    faulted: it reports whatever torch already has selected (0 in a fresh
    shell), and binding to a faulted chip blocks in ``py_init_device`` with no
    error.  ``TPU_VISIBLE_DEVICES`` (first entry) selects the logical chip.
    """
    return int(os.environ.get("TPU_VISIBLE_DEVICES", "0").split(",")[0])


def _causal_mask(q_len: int, kv_len: int):
    """fp32 additive [q_len, kv_len] causal mask, extend tokens at the tail."""
    import torch

    query_offset = kv_len - q_len
    pos = torch.arange(q_len).unsqueeze(1) + query_offset
    j = torch.arange(kv_len).unsqueeze(0)
    mask = torch.zeros(q_len, kv_len, dtype=torch.float32)
    mask.masked_fill_(j > pos, MASK_NEG)
    return mask


def _batched_mask(q_lens, kv_lens, q_pad, kv_pad):
    """Block-diagonal [B, q_pad, kv_pad] mask, the same contract the driver uses."""
    import torch

    masks = torch.full((len(q_lens), q_pad, kv_pad), MASK_NEG, dtype=torch.float32)
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        masks[b, :q_len, :kv_len] = _causal_mask(q_len, kv_len)
        masks[b, q_len:, :] = 0.0  # padded query rows: neutral (outputs dropped)
    return masks


def _build_token_index(req_to_token, q_lens, kv_lens, hkv, kv_pad):
    """[B, Hkv, kv_pad] int32 of ``slot * Hkv + h``, padded columns at slot 0.

    Mirrors ``TpuAttnBackend._build_paged_token_index`` so the test exercises
    the same construction the driver uses.
    """
    import torch

    slots = torch.zeros((len(q_lens), kv_pad), dtype=torch.int32)
    for b, kv_len in enumerate(kv_lens):
        slots[b, :kv_len] = req_to_token[b, :kv_len]
    heads = torch.arange(hkv, dtype=torch.int32)
    return (slots.unsqueeze(1) * hkv + heads.view(1, -1, 1)).contiguous()


def _ref_attention(q, k_cache, v_cache, req_to_token, q_lens, kv_lens, scale):
    """Torch reference: gather each request's K/V rows, then per-head softmax.

    ``q`` is the packed ``[T, Hq, D]`` query buffer (CPU), the caches are the
    NHD ``[Slots, Hkv, D]`` buffers (CPU).
    """
    import torch

    hq = q.shape[1]
    hkv = k_cache.shape[1]
    head_rep = hq // hkv
    out = torch.empty(q.shape[0], hq, q.shape[-1], dtype=torch.float32)
    q_start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        tokens = req_to_token[b, :kv_len].long()
        k_i = k_cache[tokens].float()  # [kv_len, Hkv, D]
        v_i = v_cache[tokens].float()
        mask = _causal_mask(q_len, kv_len)
        q_i = q[q_start : q_start + q_len].float()  # [q_len, Hq, D]
        for h in range(hq):
            kv_h = h // head_rep
            scores = (q_i[:, h, :] @ k_i[:, kv_h, :].t()) * scale + mask
            p = torch.softmax(scores, dim=-1)
            out[q_start : q_start + q_len, h, :] = p @ v_i[:, kv_h, :]
        q_start += q_len
    return out


def _run_test(
    kernel, q_lens, kv_lens, HQ, HKV, D, SCALE, rows, do_profile, do_verbose,
    dtype, block_m=64, block_n=256,
):
    import torch

    dev = f"tpu:{_logical_device()}"
    torch.manual_seed(0)

    q_pad = -(-q_lens[0] // block_m) * block_m
    kv_pad = max(q_pad, -(-kv_lens[0] // block_n) * block_n)
    # One bucket per launch: every request must round to the same padded shape.
    assert all(-(-x // block_m) * block_m == q_pad for x in q_lens)
    assert all(max(q_pad, -(-x // block_n) * block_n) == kv_pad for x in kv_lens)

    n_req = len(q_lens)
    total_q = sum(q_lens)

    # A cache with scattered slots: request b's keys are a random permutation of
    # the pool, so a wrong row index shows up immediately.
    k_cache = torch.randn(rows, HKV, D, dtype=dtype)
    v_cache = torch.randn(rows, HKV, D, dtype=dtype)
    req_to_token = torch.zeros((n_req, max(kv_lens)), dtype=torch.int32)
    for b in range(n_req):
        req_to_token[b, : kv_lens[b]] = torch.randperm(rows)[: kv_lens[b]].int()

    q = torch.randn(total_q, HQ, D, dtype=dtype)
    q_off = torch.tensor(
        [sum(q_lens[:b]) for b in range(n_req)], dtype=torch.int32
    )

    idx = _build_token_index(req_to_token, q_lens, kv_lens, HKV, kv_pad)
    mask = _batched_mask(q_lens, kv_lens, q_pad, kv_pad).contiguous()

    # The kernel reads each request's query tile at its true offset with no
    # clamp, so the last request's trailing tile over-reads past ``total_q``.
    # Pad Q up to the largest row any tile reads (``max(q_off) + q_pad``), the
    # same bound the driver pads to; the extra rows feed only discarded output.
    max_read = int(q_off.max().item()) + q_pad
    q_dev_src = q
    if total_q < max_read:
        pad = torch.zeros(max_read, HQ, D, dtype=dtype)
        pad[:total_q] = q
        q_dev_src = pad

    k_dev = k_cache.to(dev)
    v_dev = v_cache.to(dev)
    q_dev = q_dev_src.to(dev)
    idx_dev = idx.to(dev)
    mask_dev = mask.to(dev)
    off_dev = q_off.to(dev)

    if do_profile:
        kernel.adapter.enable_profile()

    out = kernel(q_dev, k_dev, v_dev, idx_dev, mask_dev, off_dev)
    out = out.float().cpu()  # [B, q_pad, Hq, D]

    if do_profile:
        kernel.adapter.collect_profile(verbose=do_verbose)

    ref = _ref_attention(q, k_cache, v_cache, req_to_token, q_lens, kv_lens, SCALE)

    # Compare each request's real rows only: the padded rows are garbage by
    # contract and the driver never reads them.
    got = torch.cat(
        [out[b, :q_len] for b, q_len in enumerate(q_lens)], dim=0
    )
    diff = (got - ref).abs()
    rel = (diff.norm() / ref.norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0
    ).item()
    print(f"  dtype       : {dtype}")
    print(f"  q_lens      : {q_lens}   kv_lens: {kv_lens}")
    print(f"  padded      : q_pad={q_pad} kv_pad={kv_pad} T={total_q} B={n_req}")
    print(f"  max abs err : {diff.max().item():.6f}")
    print(f"  rel err     : {rel:.4%}")
    print(f"  cosine sim  : {cos:.6f}")
    ok = cos > 0.999 and rel < 0.02
    if not ok:
        # Localize the error: per request, then per block_M-row tile within the
        # worst request, then the worst individual rows. A row-tile that is
        # ``tq``'s slab (rows [tq*block_m, (tq+1)*block_m)) pinpoints whether the
        # fault is a specific query tile (e.g. the last partial tile) or a kv
        # block, and the worst-row list shows if it is the over-read boundary.
        def _cos(a, bq):
            return torch.nn.functional.cosine_similarity(
                a.flatten(), bq.flatten(), dim=0
            ).item()

        per_req = []
        start = 0
        for b, q_len in enumerate(q_lens):
            g = got[start : start + q_len]
            r = ref[start : start + q_len]
            per_req.append((b, _cos(g, r), (g - r).abs().max().item()))
            start += q_len
        print("  per-request [b: cos, maxabs]:")
        for b, c, m in per_req:
            flag = "  <-- worst" if b == min(per_req, key=lambda t: t[1])[0] else ""
            print(f"      req {b} (q_len={q_lens[b]}, kv_len={kv_lens[b]}): "
                  f"cos={c:.5f} maxabs={m:.5f}{flag}")

        wb = min(per_req, key=lambda t: t[1])[0]
        wstart = sum(q_lens[:wb])
        wg = got[wstart : wstart + q_lens[wb]]
        wr = ref[wstart : wstart + q_lens[wb]]
        print(f"  worst request {wb} per {block_m}-row tile:")
        for tq in range((q_lens[wb] + block_m - 1) // block_m):
            lo, hi = tq * block_m, min((tq + 1) * block_m, q_lens[wb])
            print(f"      tile tq={tq} rows[{lo}:{hi}]: cos={_cos(wg[lo:hi], wr[lo:hi]):.5f}")
        row_err = (wg - wr).abs().amax(dim=(1, 2))  # [q_len] worst over H,D per row
        worst_rows = torch.topk(row_err, min(8, row_err.numel())).indices.tolist()
        print(f"      worst rows (by max abs err): "
              f"{sorted(worst_rows)}  (q_len={q_lens[wb]})")
    if ok:
        print("PASS: TPU paged flash attention matches torch reference")
        return True
    print("FAIL")
    return False


def _build_kernel(d, scale, dtype):
    return flash_attention_paged(d=d, sm_scale=scale, dtype=str(dtype).replace("torch.", ""))


def main():
    import torch

    do_profile = "--profile" in sys.argv
    do_verbose = "--verbose" in sys.argv
    # --profile implies --run: profiling a kernel that never executed is moot.
    do_run = "--run" in sys.argv or do_profile

    if do_run:
        # Must precede the first device tensor: torch_tpu's lazy init binds to
        # the *current* device, and a faulted chip blocks in py_init_device with
        # no error (chip 0 is faulted on this machine).
        import torch_tpu  # noqa: F401

        torch.tpu.set_device(_logical_device())
        print(f"logical device {torch.tpu.current_device()}")

    D = 128
    HQ, HKV = 16, 8  # Qwen3-0.6B GQA ratio 2:1
    SCALE = float(D) ** -0.5

    kernel = _build_kernel(D, SCALE, torch.float16)
    print("func_source:")
    print(flash_attention_paged.func_source)
    print("signature:")
    print(flash_attention_paged.signature)
    print("TIR:")
    print(flash_attention_paged.get_tir().script())
    print("\nTPU paged flash-attention kernel source:")
    print(kernel.get_kernel_source())
    print("\nPaged flash-attention compilation for TPU target succeeded.")

    if not do_run:
        return

    # (q_lens, kv_lens, rows) — one bucket each, ragged within the bucket.
    #   1. prefill, uniform            2. prefill, ragged lengths in one bucket
    #   3. decode bucket               4. mixed short/long prefill
    #
    # Every request must have kv_len >= q_len. This is the sglang invariant
    # (kv_len = prefix_len + extend_len, q_len = extend_len), so query_offset =
    # kv_len - q_len >= 0 and row 0 always attends at least key 0.  A case with
    # kv_len < q_len makes the leading rows attend *zero* keys (all columns
    # masked); kernel and torch then both fall back to a uniform average but over
    # different column counts (the kernel includes the padded gather columns),
    # which is a meaningless mismatch, not a kernel bug.
    CASES = [
        ([128] * 8, [256] * 8, 2048),
        ([128, 100, 65, 70], [256, 200, 130, 100], 2048),
        ([1, 1, 1, 1], [512, 300, 257, 400], 2048),
        ([192, 130, 191, 129], [512, 300, 500, 257], 4096),
    ]
    for _q, _kv, _ in CASES:
        assert all(kv >= q for q, kv in zip(_q, _kv)), (
            f"invalid case: kv_len < q_len in {list(zip(_q, _kv))}"
        )

    ok = True
    for dtype in (torch.float16, torch.bfloat16):
        k = _build_kernel(D, SCALE, dtype)
        print(f"\n=== {dtype} ===")
        for q_lens, kv_lens, rows in CASES:
            ok &= _run_test(
                k, q_lens, kv_lens, HQ, HKV, D, SCALE, rows,
                do_profile, do_verbose, dtype=dtype,
            )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
