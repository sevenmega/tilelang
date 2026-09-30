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

def _run_test(kernel, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose):
    import torch

    torch.manual_seed(0)
    q = torch.randn(B, HQ, SQ, D, dtype=torch.float16)
    k = torch.randn(B, HKV, SKV, D, dtype=torch.float16)
    v = torch.randn(B, HKV, SKV, D, dtype=torch.float16)
    mask2d = _build_causal_mask(SQ, SKV)  # [Sq, Skv]
    mask = mask2d.unsqueeze(0).expand(B, SQ, SKV).contiguous()  # [B, Sq, Skv]

    # Profiling must be armed before the first launch; the dump is collected
    # after the run below (same as the GEMM tests).
    if do_profile:
        kernel.adapter.enable_profile()

    out = kernel(q, k, v, mask)  # [B, Hq, Sq, D] fp16
    out = out.float()
    ref = _ref_attention(q, k, v, mask, SCALE)

    if do_profile:
        kernel.adapter.collect_profile(verbose=do_verbose)

    diff = (out - ref).abs()
    rel = (diff.norm() / ref.norm().clamp_min(1e-6)).item()
    cos = torch.nn.functional.cosine_similarity(
        out.flatten(), ref.flatten(), dim=0
    ).item()
    print(f"  shape       : tpu {tuple(out.shape)}  ref {tuple(ref.shape)}")
    print(f"  max abs err : {diff.max().item():.6f}")
    print(f"  rel err     : {rel:.4%}")
    print(f"  cosine sim  : {cos:.6f}")
    if cos > 0.999 and rel < 0.02:
        print("PASS: TPU flash attention matches torch reference")
    else:
        print("FAIL")
        sys.exit(1)


def main():
    D= 128
    SCALE = float(D) ** -0.5  # 1/sqrt(d) — matches the kernel's default sm_scale

    # Dynamic-shape kernel: only D (head dim) is baked; B/Hq/Sq/Hkv/Skv are
    # resolved at call time from the input shapes. Calling the jit factory
    # (lazy mode) returns the compiled kernel.
    kernel = flash_attention_gqa(d=D, sm_scale=SCALE)

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

    # 1. Qwen3-0.6B attention head config; tile-aligned prefill shapes, B=1.
    B, SQ, SKV, HQ, HKV = 1, 8, 128, 16, 8
    _run_test(kernel, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose)

    # 2. Qwen3-0.6B attention head config; tile-aligned prefill shapes, batched.
    B, SQ, SKV, HQ, HKV = 4, 256, 256, 16, 8
    _run_test(kernel, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose)

    # 3. Qwen3-0.6B attention head config; tile-aligned decode shapes, B=1.
    B, SQ, SKV, HQ, HKV = 1, 1, 384, 16, 8
    _run_test(kernel, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose)

    # 4. Qwen3-0.6B attention head config; tile-aligned decode shapes, batched.
    B, SQ, SKV, HQ, HKV = 4, 1, 384, 16, 8
    _run_test(kernel, B, SQ, SKV, HQ, HKV, D, SCALE, do_profile, do_verbose)

if __name__ == "__main__":
    main()

