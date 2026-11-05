# Per-call check of g4fast.moe against vLLM's FLASHINFER_CUTLASS MoE on real decode inputs (eager mode):
# G4F_MOE=1 G4F_MOE_CHECK=1 runs both and prints the relative L2 difference of each layer's MoE output.
import os, sys
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
llm = LLM(model="nvidia/Gemma-4-26B-A4B-NVFP4", revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
          tokenizer_revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6", seed=0, max_model_len=4096,
          gpu_memory_utilization=0.85, enable_prefix_caching=False, language_model_only=True, enforce_eager=True)
o = llm.chat([{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
             SamplingParams(max_tokens=8, temperature=0), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
print("TEXT", repr(o[0].outputs[0].text))
