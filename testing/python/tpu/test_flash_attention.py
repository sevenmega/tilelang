"""Standalone flash-attention (online softmax, GQA) TPU kernel test.

Compiles the **real tilelang DSL kernel** ``flash_attention_gqa`` (see
:mod:`tilelang.tpu.kernels.attention`) through the *generic* TIR->PPL translator
(``.compile(...)`` -> :mod:`tilelang.tpu.ppl_codegen`) — there is no bespoke
attention template.  The output is compared against a plain torch per-head
softmax reference (GQA grouping + additive mask).

The DSL kernel uses fixed on-chip tiles ``block_M=8`` / ``block_N=128`` and does
not handle partial tiles, so the test shapes use ``Sq`` a multiple of 8 and
``Skv`` a multiple of 128.  Layout is head-major:

    Q, Out : [B, Hq, Sq, D]      (fp16)
    K, V   : [B, Hkv, Skv, D]    (fp16)
    Mask   : [B, Sq, Skv]        (fp32 additive)

Run on real hardware under the PPL env (see the ``ppl-tpu-run-recipe`` memo):

    python testing/python/tpu/test_flash_attention.py --run

Without ``--run`` it only emits + prints the kernel source (compile smoke test).

Add ``--profile`` to enable hardware profiling of the run (implies ``--run``);
``--verbose`` forwards to ``bigTpuProfile`` for per-op detail.  Same convention
as the GEMM tests (``test_gemm_naive_tpu.py`` / ``test_gemm_v1_dyn_shape_tpu.py``):

    python testing/python/tpu/test_flash_attention.py --run --profile [--verbose]
"""

import sys

from tilelang.tpu.kernels.attention import flash_attention_gqa

MASK_NEG = -30000.0  # fp16-safe stand-in for -inf

def _build_causal_mask(sq, skv):
    import torch

    query_offset = skv - sq
    i = torch.arange(sq).unsqueeze(1)  # [sq, 1]
    j = torch.arange(skv).unsqueeze(0)  # [1, skv]
    disallowed = j > (i + query_offset)
    mask = torch.zeros(sq, skv, dtype=torch.float32)
    mask.masked_fill_(disallowed, MASK_NEG)
    return mask  # [sq, skv]


def _ref_attention(q, k, v, mask, scale):
    """Torch per-head reference. q [B,Hq,Sq,D], k/v [B,Hkv,Skv,D], mask [B,Sq,Skv]."""
    import torch

    b, hq, sq, d = q.shape
    _, hkv, skv, _ = k.shape
    head_rep = hq // hkv
    out = torch.empty(b, hq, sq, d, dtype=torch.float32)
    for bi in range(b):
        for h in range(hq):
            kv_h = h // head_rep
            qi = q[bi, h, :, :].float()  # [Sq, D]
            ki = k[bi, kv_h, :, :].float()  # [Skv, D]
            vi = v[bi, kv_h, :, :].float()  # [Skv, D]
            scores = (qi @ ki.t()) * scale + mask[bi]  # [Sq, Skv]
            p = torch.softmax(scores, dim=-1)
            out[bi, h, :, :] = p @ vi
    return out

def _run_test(
    kernel, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose,
    dtype=None, block_m=64, block_n=256,
):
    import torch

    from tilelang.tpu.ppl_runner import ensure_tpu_device

    if dtype is None:
        dtype = torch.float16
    dev = f"tpu:{ensure_tpu_device()}"
    torch.manual_seed(0)

    # The kernel has baked tiles and does not clamp partial tiles, so every
    # loaded tile must be fully in range. Pad exactly as the sglang path does
    # (_flash_pad_lens): Sq up to block_M, Skv up to block_N, kv_pad >= q_pad.
    q_pad = -(-SQ // block_m) * block_m
    kv_pad = max(q_pad, -(-SKV // block_n) * block_n)

    q = torch.randn(B, HQ, SQ, D, dtype=dtype, device=dev)
    k = torch.randn(B, HKV, SKV, D, dtype=dtype, device=dev)
    v = torch.randn(B, HKV, SKV, D, dtype=dtype, device=dev)
    qp = torch.zeros(B, HQ, q_pad, D, dtype=dtype, device=dev)
    kp = torch.zeros(B, HKV, kv_pad, D, dtype=dtype, device=dev)
    vp = torch.zeros(B, HKV, kv_pad, D, dtype=dtype, device=dev)
    qp[:, :, :SQ] = q
    kp[:, :, :SKV] = k
    vp[:, :, :SKV] = v

    mask2d = _build_causal_mask(SQ, SKV)  # [Sq, Skv] (host)
    # Padded key columns are masked so no real query attends them; padded query
    # rows are 0 (their outputs are sliced off, and an all-(-30000) row would
    # perturb the tile-wide softmax max).
    padded = torch.full((B, q_pad, kv_pad), MASK_NEG, dtype=torch.float32)
    padded[:, :SQ, :SKV] = mask2d
    padded[:, SQ:, :] = 0.0
    mask = padded.to(dev)

    # Profiling must be armed before the first launch; the dump is collected
    # after the run below (same as the GEMM tests).
    if do_profile:
        kernel.adapter.enable_profile()

    out = kernel(qp, kp, vp, mask)  # [B, Hq, q_pad, D], kernel dtype, on device
    # The torch reference is a CPU computation (on-device matmul is not a valid
    # oracle here), so move the kernel result to host for the comparison.
    out = out.float().cpu()[:, :, :SQ, :]
    ref = _ref_attention(
        q.cpu(), k.cpu(), v.cpu(), mask2d.unsqueeze(0).expand(B, SQ, SKV), SCALE
    )

    if do_profile:
        kernel.adapter.collect_profile(verbose=do_verbose)

    diff = (out - ref).abs()
    rel = (diff.norm() / ref.norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0
    ).item()
    print(f"  dtype       : {dtype}")
    print(f"  shape       : tpu {tuple(out.shape)}  ref {tuple(ref.shape)}")
    print(f"  max abs err : {diff.max().item():.6f}")
    print(f"  rel err     : {rel:.4%}")
    print(f"  cosine sim  : {cos:.6f}")
    if cos > 0.999 and rel < 0.02:
        print("PASS: TPU flash attention matches torch reference")
        return True
    print("FAIL")
    return False


def _build_kernel(d, scale, dtype):
    """Compile one shape-generic kernel for ``dtype`` (fp16 / bf16)."""
    return flash_attention_gqa(
        d=d,
        sm_scale=scale,
        # Must match the tensors handed to the kernel. The kernel is compiled
        # per dtype (the PPL copy path has no bf16<->fp16 pair), so leaving
        # this at the fp16 default while feeding bf16 silently reinterprets
        # the bits and produces garbage.
        dtype=str(dtype).replace("torch.", ""),
    )


def main():
    import torch

    D = 128
    SCALE = float(D) ** -0.5  # 1/sqrt(d) — matches the kernel's default sm_scale

    # Dynamic-shape kernel: only D (head dim), scale and dtype are baked;
    # B/Hq/Sq/Hkv/Skv are resolved at call time from the input shapes.
    kernel = _build_kernel(D, SCALE, torch.float16)

    print("func_source:")
    print(flash_attention_gqa.func_source)
    print("signature:")
    print(flash_attention_gqa.signature)
    print("TIR:")
    print(flash_attention_gqa.get_tir().script())

    print("\nTPU flash-attention kernel source:")
    print(kernel.get_kernel_source())
    print("\nFlash-attention compilation for TPU target succeeded.")

    do_profile = "--profile" in sys.argv
    do_verbose = "--verbose" in sys.argv

    # --profile implies --run: profiling a kernel that never executed is moot.
    if "--run" not in sys.argv and not do_profile:
        return

    # Qwen3-0.6B head config. Shapes are the *logical* ones; _run_test pads to
    # the kernel tiles (block_M=64, block_N=256), so unaligned lengths are fine.
    #   1. prefill, B=1        2. prefill, batched
    #   3. decode,  B=1        4. decode,  batched
    SHAPES = [
        (1, 8, 128, 16, 8),
        (32, 256, 256, 16, 8),
        (1, 1, 384, 16, 8),
        (32, 1, 384, 16, 8),
    ]

    ok = True
    for dtype in (torch.float16, torch.bfloat16):
        k = _build_kernel(D, SCALE, dtype)
        print(f"\n=== {dtype} ===")
        for B, SQ, SKV, HQ, HKV in SHAPES:
            ok &= _run_test(
                k, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose,
                dtype=dtype,
            )
    if not ok:
        sys.exit(1)

if __name__ == "__main__":
    main()

