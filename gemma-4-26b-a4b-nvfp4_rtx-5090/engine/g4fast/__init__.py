"""vLLM general plugin: exact decode fast paths, switched by environment variables.

G4F_SAMPLER=1  top-k-restricted seeded sampler (same tokens as vLLM's Triton top-k/top-p + Gumbel path).
G4F_MOE=1      batch-1 NVFP4 MoE decode kernels (same weights and NVFP4 activation scheme).
G4F_HEAD=1     Triton output layer, exact bound-pruned at batch 1 for top-k sampling.
G4F_PACK=1     batch-1 qkv/o/gate_up GEMVs from losslessly packed BF16 weights (needs a fresh torch.compile graph).
"""
import os

_done = False


def register():
    global _done
    if _done:
        return
    _done = True
    if os.environ.get("G4F_SAMPLER", "0") == "1":
        from . import sampler
        sampler.install()
    if os.environ.get("G4F_MOE", "0") == "1":
        from . import moe
        moe.install()
    if os.environ.get("G4F_HEAD", "0") == "1":
        from . import head
        head.install()
    if os.environ.get("G4F_PACK", "0") == "1":
        from . import packed
        packed.install()
