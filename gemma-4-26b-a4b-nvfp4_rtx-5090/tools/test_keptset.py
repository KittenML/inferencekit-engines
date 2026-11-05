# Kept-set check: vLLM 0.31.0's apply_top_k_top_p mask (logit > -inf after it) against the rule the
# g4fast sampler implements (exactly k tokens in (logit desc, id asc) order, then top-p while the mass
# ranked before a token is < p), on rows with heavy ties at the k-th value and at the top-p boundary.
import sys, torch
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
dev = "cuda"; V = 262144
torch.manual_seed(1)
def rule(x, k, p):
    order = sorted(range(V), key=lambda i: (-x[i], i))  # slow but exact; x is a python list
    kept = order[:k]
    v = torch.tensor([x[i] for i in kept], dtype=torch.float32)
    probs = torch.softmax(v, 0)
    excl = torch.cumsum(probs, 0) - probs
    if p >= 1.0:  # top_p 1.0: no top-p step (vLLM passes None, the sampler compiles HAS_P=False)
        return set(kept)
    return set(i for i, e in zip(kept, excl.tolist()) if e < p)
cases = 0; diff = 0; diff_k = 0
for trial in range(12):
    kind = trial % 4
    if kind == 0:   # coarse grid: many ties at the k-th value
        x = (torch.randn(V, device=dev) * 2).mul(4).round().div(4); x[7] = 30
    elif kind == 1:  # softcapped bf16 logits
        x = (torch.tanh(torch.randn(V, device=dev).mul(6) / 30) * 30).to(torch.bfloat16).float()
    elif kind == 2:  # a big tie group straddling the k-th position
        x = torch.randn(V, device=dev) - 10; x[:40] = 5.0; x[1000:1100] = 4.0
    else:            # ties straddling the top-p boundary
        x = torch.randn(V, device=dev) - 20; x[:3] = 3.0; x[500:520] = 1.0
    for k, p in [(64, 0.95), (64, 1.0), (20, 0.5)]:
        ref = apply_top_k_top_p(x.clone().view(1, -1), torch.tensor([k], device=dev, dtype=torch.int32),
                                torch.tensor([p], device=dev) if p < 1 else None)
        rs = set(torch.nonzero(ref.view(-1) > float("-inf")).view(-1).tolist())
        ms = rule(x.tolist(), k, p)
        cases += 1
        if rs != ms:
            diff += 1
            print(f"DIFF kind={kind} k={k} p={p}: vllm keeps {len(rs)}, rule keeps {len(ms)}; "
                  f"only vllm {sorted(rs - ms)[:8]}, only rule {sorted(ms - rs)[:8]}")
print(f"KEPTSET_TEST cases={cases} differing={diff}")
