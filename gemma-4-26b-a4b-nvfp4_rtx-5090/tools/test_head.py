# End-to-end check of g4fast.head (G4F_HEAD_CHECK=1): every step computes the full head too and checks that
# the candidate rows are bit-identical to it and that every row of the full top-k set is a candidate.
import os, json
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
llm = LLM(model="nvidia/Gemma-4-26B-A4B-NVFP4", revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6",
          tokenizer_revision="a19cfe00be84568a6867111c9a68c9c44fdcffe6", seed=0, max_model_len=8192,
          gpu_memory_utilization=0.90, enable_prefix_caching=False, language_model_only=True,
          compilation_config={"compile_sizes": [1]})
prompts = ["Write a Python function that merges two sorted lists, with tests.",
           "用中文写一首关于秋天的短诗，并解释其意象。",
           "A train leaves at 3:15 pm at 80 km/h; another at 4:00 pm at 100 km/h. When does the second catch up? Think step by step.",
           "List ten emoji that represent weather and explain each one 🌦️.",
           "hi"]
kw = dict(chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
for i, p in enumerate(prompts):
    for sp in [SamplingParams(max_tokens=200, temperature=1.0, top_k=64, top_p=0.95, seed=100 + i),
               SamplingParams(max_tokens=100, temperature=0.0, top_k=64, top_p=0.95),
               SamplingParams(max_tokens=60, temperature=1.0, top_k=1, seed=7),
               # exact length, stop tokens masked by min_tokens (as a benchmark sends them)
               SamplingParams(max_tokens=256, min_tokens=256, ignore_eos=True, temperature=1.0, top_k=64, top_p=0.95, seed=5 + i),
               SamplingParams(max_tokens=256, min_tokens=256, ignore_eos=True, temperature=0.0)]:
        o = llm.chat([{"role": "user", "content": p}], sp, **kw)
        print("OUT", i, sp.temperature, sp.top_k, len(o[0].outputs[0].token_ids), repr(o[0].outputs[0].text[:60]), flush=True)
