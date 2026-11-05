"""JIT-builds the CUDA decode kernels (torch.utils.cpp_extension.load) on first import."""
import os

import torch
from torch.utils.cpp_extension import load

_here = os.path.dirname(os.path.abspath(__file__))
_ext = None


def ops():
    global _ext
    if _ext is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        _ext = load(
            name=f"qwen3_fp8_decode_sm{arch}",
            sources=[os.path.join(_here, "decode.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            extra_cflags=["-O3"],
            verbose=os.environ.get("KERNEL_BUILD_VERBOSE") == "1",
        )
    return _ext
