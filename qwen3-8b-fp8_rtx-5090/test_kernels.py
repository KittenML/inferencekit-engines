"""Per-op checks of the CUDA decode kernels against the PyTorch/Triton reference ops.

Weights: layer 0 of the checkpoint, starting from the layout the engine actually loads (K-major tiles by
default). The reference gets the row-major matrix recovered with untile_fp8(w, m.kmajor) (checked
byte-equal to the checkpoint's matrix when the row-major prefill copy exists), and each GEMV variant gets
that matrix re-tiled in the order it is told to read (km=0 group-major, km=1 K-major).
Inputs: the GEMV and lm_head checks run on two activation vectors: "emb" (a token's embedding row, the
real input of layer 0) and "rnd" (random, N(0, 0.5^2)). Attention runs on random q/k/v and a random KV cache.
Reference for the FP8 linear layers: kernels/prefill.fp8_linear, this repo's Triton copy of the
reference recipe (same tensor-core instruction, block-scale order and division instruction as the
transformers fine-grained FP8 kernel; that kernel itself is not called here).
Pass criteria: FP8 GEMVs and the K/V cache write must be 100% bitwise identical (q's RMSNorm + RoPE is
not written out by the kernel, so it is only checked through the attention output); attention must be
within rel-L2 5e-3 of an fp32 ground truth; the lm_head is reported (1-ulp differences vs cuBLAS are
expected) and its fused argmax must equal both the argmax of its own logits and cuBLAS's argmax.
usage: python test_kernels.py --model PATH
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from generate import Engine, source_identity  # noqa: E402
from kernels.prefill import fp8_linear  # noqa: E402
from model import rms_norm, rotate_half, tile_fp8, untile_fp8  # noqa: E402


def ulp_dist(a, b):
    """Elementwise distance in bf16 units in the last place between two bf16 tensors (0 = identical bits)."""
    def line(x):  # bf16 bit patterns -> integers in value order (sign-magnitude to two's complement)
        i = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i + 32768), i)
    return (line(a) - line(b)).abs().flatten()


def ulp_summary(got, ref):
    d = ulp_dist(got, ref)
    n = int((d > 0).sum())
    if n == 0:
        return "  max ulp 0"
    i = int(d.argmax())
    return (f"  max ulp {int(d[i])} (at |ref| {abs(ref.flatten()[i].float().item()):.2e});"
            f" of the {n} differing: {int((d == 1).sum())} at 1 ulp, {int((d <= 2).sum()) - (d.numel() - n)} at <= 2")


def report(name, got, ref, exact=False):
    ulp = ulp_summary(got, ref) if got.dtype == ref.dtype == torch.bfloat16 else ""
    got, ref = got.float().flatten(), ref.float().flatten()
    same = (got == ref).float().mean().item()
    diff = (got - ref).abs().max().item()
    rel = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    ok = same == 1.0 if exact else (same > 0.99 and rel < 1e-2)
    print(f"{'OK  ' if ok else 'FAIL'} {name:40s} bitwise-equal {same * 100:7.3f}%  max|diff| {diff:.3e}  rel-l2 {rel:.2e}"
          + ulp + ("" if exact else "  (not required bitwise)"))
    return ok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    eng = Engine(args.model, max_seq=2048)
    m, k, cfg = eng.m, eng.k, eng.m.cfg
    L = m.layers[0]
    ok = True
    import triton
    print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), triton {triton.__version__}, "
          f"{torch.cuda.get_device_name(0)}; engine {source_identity()}; weights loaded with kmajor={m.kmajor}")

    # Row-major weights as the checkpoint stores them, recovered from the engine's loaded (tiled) layout.
    names = ("w_qkv", "w_o", "w_gu", "w_d")
    W = {n: untile_fp8(getattr(L, n), m.kmajor) for n in names}
    for n, r in zip(names, ("r_qkv", "r_o", "r_gu", "r_d")):
        rm = getattr(L, r, None)
        if rm is not None:   # the row-major prefill copy is the checkpoint matrix itself
            same = torch.equal(W[n].view(torch.uint8), rm.view(torch.uint8))
            print(f"{'OK  ' if same else 'FAIL'} untile({n}, kmajor={m.kmajor}) == checkpoint matrix")
            ok &= same
    for km in (False, True):   # tiling round trips in both orders
        rt = untile_fp8(tile_fp8(W["w_qkv"], km), km)
        same = torch.equal(rt.view(torch.uint8), W["w_qkv"].view(torch.uint8))
        print(f"{'OK  ' if same else 'FAIL'} tile/untile round trip kmajor={km}")
        ok &= same

    # Activations: a real one (embedding row of a common token) and a random one.
    inputs = {"emb": m.embed[9707].clone(), "rnd": (torch.randn(cfg.hidden, device="cuda") * 0.5).to(torch.bfloat16)}
    o_in = {"emb": (torch.randn(cfg.hidden, device="cuda")).to(torch.bfloat16),
            "rnd": (torch.randn(cfg.hidden, device="cuda")).to(torch.bfloat16)}

    # (depth, warps, km, pre, trig) per kind; pre/trig only change timing, not numerics. The last variant is
    # exactly what the engine's decode graph launches.
    variants = [("d4w4k0", [(4, 4, 0, -1, -1)] * 4), ("d8w2k0", [(8, 2, 0, -1, -1)] * 4),
                ("d4w1k0", [(4, 1, 0, -1, -1)] * 4), ("d8w8k0", [(8, 8, 0, -1, -1)] * 4),
                ("d4w4k1", [(4, 4, 1, -1, -1)] * 4), ("d8w4k1", [(8, 4, 1, -1, -1)] * 4),
                ("engine", [(eng.depth[i], eng.warps[i], int(m.kmajor), eng.pre[i], eng.trig[i]) for i in range(4)])]
    tiled = {km: {n: tile_fp8(W[n], bool(km)) for n in names} for km in (0, 1)}
    for vname, per_kind in variants:
        for iname, resid in inputs.items():
            tag = f"{vname} {iname}"
            # 0: qkv = fp8_linear(rmsnorm(resid))
            d, w, km, pre, trig = per_kind[0]
            ref = fp8_linear(rms_norm(resid[None], L.ln1, cfg.eps), W["w_qkv"], L.s_qkv)[0]
            out = torch.empty_like(ref)
            k.fp8_gemv(0, tiled[km]["w_qkv"], L.s_qkv, resid, L.ln1, out, L.w_qkv.shape[0], cfg.eps, d, w, pre, trig, km)
            ok &= report(f"gemv qkv (rmsnorm) {tag}", out, ref, exact=True)
            # 1: o_proj (+ residual), no norm in the prologue
            d, w, km, pre, trig = per_kind[1]
            a = o_in[iname]
            r0 = resid.clone()
            ref = r0 + fp8_linear(a[None], W["w_o"], L.s_o)[0]
            k.fp8_gemv(1, tiled[km]["w_o"], L.s_o, a, None, r0, L.w_o.shape[0], cfg.eps, d, w, pre, trig, km)
            ok &= report(f"gemv o_proj (+resid) {tag}", r0, ref, exact=True)
            # 2: gate|up with the SwiGLU epilogue
            d, w, km, pre, trig = per_kind[2]
            gu = fp8_linear(rms_norm(resid[None], L.ln2, cfg.eps), W["w_gu"], L.s_gu)[0]
            g, u = gu.split(cfg.inter)
            hvec = F.silu(g) * u
            out = torch.empty(cfg.inter, dtype=torch.bfloat16, device="cuda")
            k.fp8_gemv(2, tiled[km]["w_gu"], L.s_gu, resid, L.ln2, out, L.w_gu.shape[0], cfg.eps, d, w, pre, trig, km)
            ok &= report(f"gemv gate_up (swiglu) {tag}", out, hvec, exact=True)
            # 3: down (K = 12288, + residual)
            d, w, km, pre, trig = per_kind[3]
            r0 = resid.clone()
            ref = r0 + fp8_linear(hvec.contiguous()[None], W["w_d"], L.s_d)[0]
            k.fp8_gemv(3, tiled[km]["w_d"], L.s_d, hvec.contiguous(), None, r0, L.w_d.shape[0], cfg.eps, d, w, pre,
                       trig, km)
            ok &= report(f"gemv down (+resid) {tag}", r0, ref, exact=True)

    # Fused attention: q/k norm + RoPE + KV-cache write of the new token + attention over [0, pos].
    # Random q/k/v and random cached K/V (not model activations).
    for pos, chunk in ((0, 64), (63, 64), (64, 64), (777, 64), (1500, 64), (31, 32), (1500, 32), (127, 128), (1500, 128)):
        eng.state[0] = pos
        kc = torch.randn(8, 2048, 128, device="cuda").to(torch.bfloat16)
        vc = torch.randn(8, 2048, 128, device="cuda").to(torch.bfloat16)
        qkv = (torch.randn(6144, device="cuda") * 2).to(torch.bfloat16)
        kc_ref, vc_ref = kc.clone(), vc.clone()
        out = torch.empty(4096, dtype=torch.bfloat16, device="cuda")
        k.attention(qkv, L.q_norm, L.k_norm, m.cos, m.sin, eng.state, kc, vc, eng.o_part, eng.ml_part,
                    eng.counters, out, 128 ** -0.5, cfg.eps, 0, chunk)
        q, kk, v = qkv.split([4096, 1024, 1024])
        cos, sin = m.cos[pos], m.sin[pos]
        qr = rms_norm(q.view(32, 128), L.q_norm, cfg.eps)
        qr = qr * cos + rotate_half(qr) * sin
        kr = rms_norm(kk.view(8, 128), L.k_norm, cfg.eps)
        kr = kr * cos + rotate_half(kr) * sin
        kc_ref[:, pos] = kr
        vc_ref[:, pos] = v.view(8, 128)
        ok &= report(f"k cache write (k norm+RoPE) pos={pos}", kc[:, pos], kr, exact=True)
        ok &= report(f"v cache write pos={pos}", vc[:, pos], v.view(8, 128), exact=True)
        untouched = bool(torch.equal(kc, kc_ref)) and bool(torch.equal(vc, vc_ref))
        if not untouched:
            print(f"FAIL attention pos={pos} wrote outside its KV slot")
        ok &= untouched
        ref = F.scaled_dot_product_attention(qr.view(1, 32, 1, 128), kc_ref[None, :, : pos + 1],
                                             vc_ref[None, :, : pos + 1], enable_gqa=True)
        # fp32 ground truth for the attention itself (our kernel keeps P in fp32; SDPA rounds P to bf16)
        ref32 = F.scaled_dot_product_attention(qr.view(1, 32, 1, 128).float(), kc_ref[None, :, : pos + 1].float(),
                                               vc_ref[None, :, : pos + 1].float(), enable_gqa=True)
        e_ours = ((out.float() - ref32.view(-1)).norm() / ref32.norm()).item()
        e_sdpa = ((ref.float() - ref32).norm() / ref32.norm()).item()
        a_ok = e_ours < 5e-3
        print(f"{'OK  ' if a_ok else 'FAIL'} attention chunk {chunk} ctx={pos + 1}: rel-l2 vs fp32  ours {e_ours:.2e}  "
              f"bf16 SDPA {e_sdpa:.2e}")
        ok &= a_ok
    counters_ok = int(eng.counters.abs().sum()) == 0
    print(f"{'OK  ' if counters_ok else 'FAIL'} attention split counters reset")
    ok &= counters_ok

    # lm_head + fused argmax (final RMSNorm prologue) vs cuBLAS F.linear
    for iname, resid in inputs.items():
        eng.best.zero_()
        k.lm_head_argmax(m.lm_head, resid, m.norm, eng.logits, eng.best, cfg.eps, eng.lm_head_ctas)
        ref = F.linear(rms_norm(resid[None], m.norm, cfg.eps), m.lm_head)[0]
        report(f"lm_head logits {iname}", eng.logits, ref)
        tok = 0xFFFFFFFF - (int(eng.best.item()) & 0xFFFFFFFF)
        own, cub = int(eng.logits.float().argmax()), int(ref.float().argmax())
        a_ok = tok == own == cub
        print(f"{'OK  ' if a_ok else 'FAIL'} lm_head argmax {iname}: fused kernel {tok}, its logits {own}, cuBLAS {cub}")
        ok &= a_ok

    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
