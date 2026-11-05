# Exactness test for g4fast.sampler.fast_sample against vLLM 0.31.0's own seeded path
# (apply_top_k_top_p + gumbel_sample): identical token ids over many seeds, positions and logit shapes.
import sys, torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from g4fast.sampler import fast_sample

torch.manual_seed(0)
dev = "cuda"
V = 262144
def case(kind, B):
    if kind == "peaked":
        x = torch.randn(B, V, device=dev) * 3; x[:, :50] += 12
    elif kind == "flat":
        x = torch.randn(B, V, device=dev) * 0.5
    elif kind == "ties":   # bf16-like coarse grid: many exact ties around the k-th value
        x = (torch.randn(B, V, device=dev) * 2).mul(4).round().div(4); x[:, 7] = 30
    elif kind == "clustered":  # top tokens packed in one block: forces the full-vocabulary pass
        x = torch.randn(B, V, device=dev); x[:, 5000:5100] += 8
    elif kind == "equal":
        x = torch.zeros(B, V, device=dev)
    elif kind == "softcap":
        x = (torch.tanh(torch.randn(B, V, device=dev).mul(6) / 30) * 30).to(torch.bfloat16).float()
    return x
n = bad = fallback = 0
for kind in ["peaked", "flat", "ties", "softcap", "clustered", "equal"]:
    for B in [1, 3]:
        for trial in range(25):
            x = case(kind, B)
            k = torch.tensor([64, 20, 1][:B], device=dev, dtype=torch.int32)
            p = torch.tensor([0.95, 0.5, 1.0][:B], device=dev, dtype=torch.float32)
            temp = torch.tensor([1.0, 0.7, 1.3, 0, 0], device=dev)
            seeds = torch.randint(0, 2**62, (5,), device=dev, dtype=torch.int64)
            idx = torch.arange(B, device=dev, dtype=torch.int32)
            pos = torch.randint(0, 8000, (B,), device=dev, dtype=torch.int64)
            ref = gumbel_sample(apply_top_k_top_p(x.clone(), k, p), idx, temp, seeds, pos,
                                apply_temperature=False, is_drafting=False)
            got = fast_sample(x.clone(), k, p, idx, temp, seeds, pos)
            if got is None:
                fallback += 1; continue
            n += B; bad += int((ref.view(-1) != got.view(-1)).sum())
            if (ref.view(-1) != got.view(-1)).any() and bad < 5:
                print("MISMATCH", kind, B, ref.tolist(), got.tolist())
# timing
x = case("softcap", 1); k = torch.tensor([64], device=dev, dtype=torch.int32); p = torch.tensor([0.95], device=dev)
temp = torch.ones(5, device=dev); seeds = torch.zeros(5, device=dev, dtype=torch.int64)
idx = torch.zeros(1, device=dev, dtype=torch.int32); pos = torch.zeros(1, device=dev, dtype=torch.int64)
def t(f, r=20):
    f(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(r): f()
    g.replay(); torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record(); g.replay(); e.record()
    torch.cuda.synchronize(); return s.elapsed_time(e) / r * 1000
print("ref us", t(lambda: gumbel_sample(apply_top_k_top_p(x.clone(), k, p), idx, temp, seeds, pos, apply_temperature=False, is_drafting=False)))
print("fast us", t(lambda: fast_sample(x, k, p, idx, temp, seeds, pos)))
print(f"SAMPLER_TEST tokens={n} mismatches={bad} fallbacks={fallback}")
sys.exit(1 if bad else 0)
