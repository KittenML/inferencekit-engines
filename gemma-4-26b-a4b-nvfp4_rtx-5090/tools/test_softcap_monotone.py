# The pruned head assumes f(z) = tanh(z / 30) * 30, computed in bf16 by torch exactly as LogitsProcessor does,
# is monotone non-decreasing over bf16 inputs. Check every finite bf16 value on the GPU.
import torch
b = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
b = b[torch.isfinite(b)]
x = b.float().sort().values.to(torch.bfloat16)
y = torch.tanh(x / 30.0) * 30.0
bad = int((y[1:].float() < y[:-1].float()).sum())
print(f"SOFTCAP_MONOTONE values={x.numel()} decreasing_steps={bad}")
