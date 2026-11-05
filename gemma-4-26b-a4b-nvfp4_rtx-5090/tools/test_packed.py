# Packed GEMV vs F.linear on every qkv/o/gate_up weight of the checkpoint: storage round trip (bit for bit, in
# _pack) and the GEMV against fp64 (error vs the BF16 GEMV's own error), with real-scale inputs.
import glob, torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
import g4fast.packed as P
root = snapshot_download("nvidia/Gemma-4-26B-A4B-NVFP4", revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
                         allow_patterns=["*.safetensors"])
n = ok = 0; worst = 0.0
for f in sorted(glob.glob(root + "/*.safetensors")):
    with safe_open(f, "pt", device="cuda") as s:
        for k in s.keys():
            if "language_model.layers" not in k or not k.endswith(".weight"):
                continue
            name = k.rsplit(".", 2)[-2]
            if name not in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "qkv_proj", "gate_up_proj"):
                continue
            w = s.get_tensor(k)
            if w.dtype != torch.bfloat16 or w.shape[1] % 256 or w.shape[0] % 8:
                continue
            P._packs.clear(); p = P._pack(w); n += 1
            if p is None:
                print("ROUNDTRIP FAIL", k); continue
            ok += 1
            x = torch.randn(1, w.shape[1], device="cuda").to(torch.bfloat16)
            y = P.packed_linear(x, w).float()
            ref = (w.double() @ x.double().view(-1)).view(1, -1)
            yl = torch.nn.functional.linear(x, w).float()
            e1 = float((y.double() - ref).abs().max()); e0 = float((yl.double() - ref).abs().max())
            worst = max(worst, e1 / max(e0, 1e-30))
            if e1 > 2 * e0 + 1e-6:
                print('BAD', k, tuple(w.shape), 'packed err', e1, 'linear err', e0, 'esc', int(p[3][-1]), 'ME', p[6], 'ymax', float(ref.abs().max()))
print(f"PACKED_TEST matrices {n}, round trip ok {ok}; max (packed error / F.linear error) vs fp64 = {worst:.2f}")
