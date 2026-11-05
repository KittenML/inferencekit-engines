"""OpenAI-compatible streaming HTTP server for the batch-1 engine, so a client-side timer
(`inferencekit bench serve`) can measure it like any other engine.

  python serve.py --model PATH [--host 127.0.0.1] [--port 8000] [--served-model-name NAME]
                  [--max-seq 4096] [--no-pdl] [--no-prefill-copy] [--no-prefill-graphs]
                  [--logprobs] [--lookahead 4] [--max-queue 0]

Standard library only (http.server); the engine's own dependencies (torch, triton, safetensors) plus
transformers for the tokenizer and chat template, which the engine's HF reference already uses.

Scheduling: batch 1. One worker thread owns the GPU and runs one request at a time; other requests
wait in a FIFO queue (`--max-queue N` answers 503 beyond N waiting requests; 0 = unbounded).

Endpoints
  GET  /health                 {"status": "ok", "waiting": n, "running": bool, "completed": n}
  GET  /v1/models              the served model, with max_model_len
  GET  /engine/config          the resolved configuration (same JSON as the startup line)
  POST /v1/chat/completions    messages -> chat template of the model's tokenizer -> tokens
  POST /v1/completions         prompt: a string, a list of token ids, or a 1-element list of either
  POST /tokenize               {"prompt": str} or {"messages": [...]} -> {"tokens", "count", "max_model_len"}
  POST /detokenize             {"tokens": [...]} -> {"prompt": str}

Generation: greedy only. `temperature` must be sent and must be 0; anything this server does not
implement is answered with HTTP 400 naming the field, never silently ignored (see `_check_sampling`).
top_p / top_k / min_p / seed are accepted because greedy argmax is invariant to them.
`max_tokens` (or `max_completion_tokens`) defaults to max_seq - prompt_len; prompt_len + max_tokens
must fit in --max-seq. `ignore_eos: true` generates exactly max_tokens tokens; `stop_token_ids`
(a vLLM extension) adds stop tokens, which apply even with ignore_eos (as in vLLM). Stop strings are
not implemented (400). `min_tokens` (vLLM/SGLang/TRT-LLM) is accepted only where it masks nothing:
ignore_eos: true, no stop_token_ids and min_tokens <= max_tokens, which is what `bench serve
--engine-kind vllm` sends (see `_check_min_tokens`); any other non-zero min_tokens is 400.

Streaming (stream: true): Server-Sent Events, `data: <json>\\n\\n`, terminated by `data: [DONE]`.
  - Exactly one chunk per generated token, sent as soon as that token is on the host. The first
    token's chunk also carries delta.role = "assistant" (chat): no role-only chunk is sent before the
    first token, so the first SSE event is the first token (time to first token = time to first event).
  - finish_reason ("stop" for a stop token, "length" for max_tokens) is set on the last token's chunk.
  - Extension fields (vLLM's `return_token_ids` names; omitted when the request sends
    return_token_ids: false): choices[0].token_ids = [the token's id] in every chunk, and
    prompt_token_ids (top level for chat, in the choice for completions) in the first chunk.
    A stop token (EOS) is generated, so it is counted in token_ids and usage; its text is empty
    when skip_special_tokens is true (the default).
  - stream_options.include_usage: every chunk carries "usage": null and a final chunk with
    "choices": [] carries usage {prompt_tokens, completion_tokens, total_tokens}.
  - The last chunk with content in it (the usage chunk if requested, else the last token chunk) also
    carries "inferencekit": {"queue_ms", "prefill", "prefill_impl", "prompt_tokens"}; "prefill" is how
    this request's prefill ran: "eager", "graph_capture" (second sighting of this prompt length: the
    capture cost is inside this request) or "graph_replay" (see DESIGN.md and Engine.prefill).
  - logprobs (chat: logprobs + top_logprobs 0 or 1; completions: logprobs 0 or 1) need --logprobs at
    startup, which adds a log-softmax over the vocabulary to every prefill (inside its CUDA graph) and,
    for requests that ask, after every decode step (outside the decode graph). The logprob is the
    chosen token's log-softmax of the engine's bf16 logits, computed in fp32. Greedy means the top-1
    alternative is the chosen token itself. Without --logprobs such requests get 400.

Startup is bench.py's: Engine(...) with the same flags, then eager generations that compile the Triton
prefill kernels and capture the decode CUDA graph. The warm-up runs one prompt per length in
--warmup-lengths (every tile shape the prefill GEMM picks), then forgets those sightings, so live
traffic starts with no prefill graph (the engine graphs a prefill on the 2nd sighting of an exact
prompt length, at most 4 lengths). It prints the resolved config to stderr:
  [serve] resolved_config {...json...}
with the settings in effect (decode graph captured, prefill graphs, kernels, KV-cache dtype, max
context, ...); `parse_config_line()` reads it back. The next line repeats the same JSON as
  {"inferencekit_resolved_config": {...json...}}
the line format inferencekit's managed server runs read from a custom engine's log.

Per token, the host overlaps with the GPU: up to --lookahead decode-graph replays are queued ahead;
after each replay an event is recorded on the compute stream and a side stream copies that step's
4-byte token id (and logprob) to pinned host memory, so the host never waits on the compute stream
itself. The compute stream sees the same graph replays as Engine.generate plus one event record per step.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_NAME = "qwen3-fp8-decode"
CONFIG_PREFIX = "[serve] resolved_config "
MAX_BODY_BYTES = 32 << 20
# One prompt per GEMM tile class of kernels/prefill.py (block_m 16/32/64/128, M % 16 == 0 or not,
# M == 1, the < 16 "ref" path) so live requests do not pay Triton compilation.
DEFAULT_WARMUP_LENGTHS = (1, 5, 16, 20, 32, 40, 64, 100, 300, 320, 500, 512, 1024, 1100)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# =================================================================================================
# Request validation (pure Python; no engine, no torch)
# =================================================================================================
class RequestError(Exception):
    def __init__(self, message, param=None, status=400, type_="invalid_request_error", code=None):
        super().__init__(message)
        self.message, self.param, self.status, self.type, self.code = message, param, status, type_, code

    def body(self):
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


_COMMON_FIELDS = {"model", "stream", "stream_options", "max_tokens", "temperature", "top_p", "top_k", "min_p",
                  "seed", "n", "ignore_eos", "stop", "stop_token_ids", "logprobs", "return_token_ids",
                  "skip_special_tokens", "user", "presence_penalty", "frequency_penalty", "repetition_penalty",
                  "logit_bias", "min_tokens"}
CHAT_FIELDS = _COMMON_FIELDS | {"messages", "max_completion_tokens", "top_logprobs", "chat_template_kwargs",
                                "add_generation_prompt"}
COMPLETION_FIELDS = _COMMON_FIELDS | {"prompt", "echo", "best_of", "suffix"}


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _bool(body, name, default):
    v = body.get(name)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise RequestError(f"{name} must be a boolean", name)
    return v


def _check_fields(body, allowed):
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise RequestError(f"unsupported field(s) {unknown}: this server rejects parameters it does not "
                           f"implement instead of ignoring them", unknown[0])


def _check_sampling(body):
    """Greedy only: reject every parameter value that would change the argmax token sequence."""
    if "temperature" not in body or body["temperature"] is None:
        raise RequestError("temperature is required and must be 0: this engine decodes greedily (argmax) "
                           "only, and the OpenAI default (1.0) means sampling", "temperature")
    t = body["temperature"]
    if not _is_num(t) or t != 0:
        raise RequestError(f"temperature must be 0 (greedy argmax only), got {t!r}", "temperature")
    # Accepted as no-ops: truncation keeps the argmax token, and greedy decoding does not use a seed.
    if body.get("top_p") is not None and not (_is_num(body["top_p"]) and 0 < body["top_p"] <= 1):
        raise RequestError("top_p must be in (0, 1]", "top_p")
    if body.get("top_k") is not None and not (_is_int(body["top_k"]) and (body["top_k"] >= 1 or body["top_k"] in (0, -1))):
        raise RequestError("top_k must be an integer >= 1, or 0 / -1 (off)", "top_k")
    if body.get("min_p") is not None and not (_is_num(body["min_p"]) and 0 <= body["min_p"] < 1):
        raise RequestError("min_p must be in [0, 1)", "min_p")
    if body.get("seed") is not None and not _is_int(body["seed"]):
        raise RequestError("seed must be an integer", "seed")
    for name, neutral in (("presence_penalty", 0), ("frequency_penalty", 0), ("repetition_penalty", 1)):
        v = body.get(name)
        if v is not None and (not _is_num(v) or v != neutral):
            raise RequestError(f"{name} is not implemented (only {neutral}, which changes nothing)", name)
    if body.get("logit_bias"):
        raise RequestError("logit_bias is not implemented", "logit_bias")
    # min_tokens is checked in parse_request: whether it is a no-op depends on the stop set and max_tokens.
    if body.get("n") not in (None, 1):
        raise RequestError("n must be 1 (one sequence per request)", "n")
    stop = body.get("stop")
    if stop not in (None, "", []):
        raise RequestError("stop strings are not implemented; use stop_token_ids", "stop")


def _check_min_tokens(min_tokens, max_tokens, stop_ids):
    """Accept min_tokens only where it cannot change the output; 400 otherwise.

    vLLM's min_tokens masks the request's stop ids (stop_token_ids, plus EOS unless ignore_eos) until
    min_tokens tokens exist. Here stop_ids is exactly that set, so when it is empty (ignore_eos: true and
    no stop_token_ids) there is nothing to mask and min_tokens <= max_tokens is a no-op. This is the body
    that `bench serve --engine-kind vllm|sglang|trtllm` sends (ignore_eos + min_tokens = max_tokens).
    With a non-empty stop set, min_tokens > 0 would mask those ids and change the argmax: not implemented."""
    if min_tokens is None:
        return
    if not _is_int(min_tokens) or min_tokens < 0:
        raise RequestError("min_tokens must be an integer >= 0", "min_tokens")
    if min_tokens > max_tokens:
        raise RequestError(f"min_tokens ({min_tokens}) must be <= max_tokens ({max_tokens})", "min_tokens")
    if min_tokens and stop_ids:
        raise RequestError("min_tokens > 0 is only accepted with ignore_eos: true and no stop_token_ids (then it "
                           "masks nothing); otherwise it would mask stop tokens, changing the argmax, which this "
                           "server does not implement", "min_tokens")


class GenRequest:
    """A validated generation request."""

    def __init__(self, kind, prompt_ids, max_tokens, stop_ids, stream, include_usage, logprobs, top_logprobs,
                 return_token_ids, skip_special_tokens, model):
        self.kind, self.prompt_ids, self.max_tokens, self.stop_ids = kind, prompt_ids, max_tokens, stop_ids
        self.stream, self.include_usage = stream, include_usage
        self.logprobs, self.top_logprobs = logprobs, top_logprobs
        self.return_token_ids, self.skip_special_tokens, self.model = return_token_ids, skip_special_tokens, model


def _message_text(m, i):
    if not isinstance(m, dict):
        raise RequestError(f"messages[{i}] must be an object", "messages")
    extra = sorted(set(m) - {"role", "content", "name"})
    if extra:
        raise RequestError(f"messages[{i}] has unsupported field(s) {extra}", "messages")
    if not isinstance(m.get("role"), str):
        raise RequestError(f"messages[{i}].role must be a string", "messages")
    c = m.get("content")
    if c is None or isinstance(c, str):
        out = dict(m)
        out["content"] = c or ""
        return out
    if isinstance(c, list):
        texts = []
        for part in c:
            if not (isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)):
                raise RequestError(f"messages[{i}].content: only text parts are supported", "messages")
            texts.append(part["text"])
        out = dict(m)
        out["content"] = "".join(texts)
        return out
    raise RequestError(f"messages[{i}].content must be a string or a list of text parts", "messages")


def parse_messages(body):
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise RequestError("messages must be a non-empty list", "messages")
    return [_message_text(m, i) for i, m in enumerate(msgs)]


def chat_prompt_ids(body, tokenizer):
    msgs = parse_messages(body)
    kwargs = body.get("chat_template_kwargs") or {}
    if not isinstance(kwargs, dict):
        raise RequestError("chat_template_kwargs must be an object", "chat_template_kwargs")
    agp = _bool(body, "add_generation_prompt", True)
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=agp, **kwargs)


def completion_prompt_ids(prompt, tokenizer):
    if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
        prompt = prompt[0]
    if isinstance(prompt, str):
        if not prompt:
            raise RequestError("prompt is empty", "prompt")
        return tokenizer.encode(prompt, add_special_tokens=True)
    if isinstance(prompt, list) and prompt and all(_is_int(t) for t in prompt):
        return list(prompt)
    if isinstance(prompt, list) and len(prompt) > 1:
        raise RequestError("one prompt per request (batch 1)", "prompt")
    raise RequestError("prompt must be a non-empty string or a non-empty list of token ids", "prompt")


def parse_request(kind, body, tokenizer, ctx):
    """kind: "chat" | "completion". ctx: served_model_name, max_seq, eos_ids, vocab_size, logprobs_supported."""
    if not isinstance(body, dict):
        raise RequestError("the request body must be a JSON object")
    _check_fields(body, CHAT_FIELDS if kind == "chat" else COMPLETION_FIELDS)
    model = body.get("model")
    if model is not None and model not in ctx["model_names"]:
        raise RequestError(f"The model `{model}` does not exist.", "model", status=404, type_="NotFoundError",
                           code="model_not_found")
    _check_sampling(body)
    stream = _bool(body, "stream", False)
    so = body.get("stream_options")
    include_usage = False
    if so is not None:
        if not stream:
            raise RequestError("stream_options is only allowed with stream: true", "stream_options")
        if not isinstance(so, dict) or set(so) - {"include_usage"}:
            raise RequestError("stream_options supports only include_usage", "stream_options")
        include_usage = _bool(so, "include_usage", False)

    if kind == "chat":
        lp = body.get("logprobs")
        if lp is not None and not isinstance(lp, bool):
            raise RequestError("logprobs must be a boolean", "logprobs")
        logprobs = bool(lp)
        tl = body.get("top_logprobs")
        if tl is not None and not (_is_int(tl) and tl in (0, 1)):
            raise RequestError("top_logprobs must be 0 or 1 (greedy: the top-1 alternative is the chosen token)",
                               "top_logprobs")
        if tl and not logprobs:
            raise RequestError("top_logprobs requires logprobs: true", "top_logprobs")
        top_logprobs = tl or 0
        mt, mct = body.get("max_tokens"), body.get("max_completion_tokens")
        if mt is not None and mct is not None and mt != mct:
            raise RequestError("max_tokens and max_completion_tokens differ", "max_tokens")
        max_tokens = mt if mt is not None else mct
    else:
        if body.get("echo"):
            raise RequestError("echo is not implemented", "echo")
        if body.get("best_of") not in (None, 1):
            raise RequestError("best_of must be 1", "best_of")
        if body.get("suffix") is not None:
            raise RequestError("suffix is not implemented", "suffix")
        lp = body.get("logprobs")
        if lp is not None and not (_is_int(lp) and lp in (0, 1)):
            raise RequestError("logprobs must be 0 or 1 (greedy: the top-1 alternative is the chosen token)",
                               "logprobs")
        logprobs = lp is not None
        top_logprobs = lp or 0
        max_tokens = body.get("max_tokens")
        if "prompt" not in body:
            raise RequestError("prompt is required", "prompt")
    if logprobs and not ctx["logprobs_supported"]:
        raise RequestError("logprobs are off on this server: start it with --logprobs", "logprobs")

    stop_tok = body.get("stop_token_ids")
    if stop_tok is not None and not (isinstance(stop_tok, list) and all(_is_int(t) for t in stop_tok)):
        raise RequestError("stop_token_ids must be a list of integers", "stop_token_ids")
    ignore_eos = _bool(body, "ignore_eos", False)
    return_token_ids = _bool(body, "return_token_ids", True)
    skip_special = _bool(body, "skip_special_tokens", True)
    if body.get("user") is not None and not isinstance(body["user"], str):
        raise RequestError("user must be a string", "user")

    prompt_ids = chat_prompt_ids(body, tokenizer) if kind == "chat" else completion_prompt_ids(body["prompt"], tokenizer)
    vocab = ctx.get("vocab_size")
    if vocab and any(not (0 <= t < vocab) for t in prompt_ids):
        raise RequestError(f"prompt token ids must be in [0, {vocab})", "prompt")
    P, max_seq = len(prompt_ids), ctx["max_seq"]
    if max_tokens is None:
        max_tokens = max_seq - P
    if not _is_int(max_tokens) or max_tokens < 1:
        raise RequestError("max_tokens must be an integer >= 1", "max_tokens")
    if P + max_tokens > max_seq:
        raise RequestError(f"This model's maximum context length is {max_seq} tokens. However, you requested "
                           f"{P + max_tokens} tokens ({P} in the messages, {max_tokens} in the completion).",
                           "max_tokens")
    stop_ids = frozenset(stop_tok or ()) | (frozenset() if ignore_eos else frozenset(ctx["eos_ids"]))
    _check_min_tokens(body.get("min_tokens"), max_tokens, stop_ids)
    return GenRequest(kind, prompt_ids, max_tokens, stop_ids, stream, include_usage, logprobs, top_logprobs,
                      return_token_ids, skip_special, model)


# =================================================================================================
# Detokenization
# =================================================================================================
class IncrementalDetokenizer:
    """Text deltas from a growing id list (the prefix/read-offset method of HF TextStreamer and vLLM).

    A token that ends inside a multi-byte UTF-8 character decodes to U+FFFD; its text is held back until
    a later token completes the character, so the deltas never contain a broken character and their
    concatenation equals decode(all ids) (flush() releases what is left at the end)."""

    def __init__(self, tokenizer, skip_special_tokens=True):
        self.tok, self.skip = tokenizer, skip_special_tokens
        self.ids, self.prefix_offset, self.read_offset = [], 0, 0

    def add(self, token_id):
        self.ids.append(token_id)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=self.skip)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=self.skip)
        if len(full) > len(prefix) and not full.endswith("�"):
            self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
            return full[len(prefix):]
        return ""

    def flush(self):
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=self.skip)
        full = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=self.skip)
        self.prefix_offset = self.read_offset = len(self.ids)
        return full[len(prefix):] if len(full) > len(prefix) else ""


# =================================================================================================
# Scheduler: one worker thread owns the model; FIFO queue; batch 1
# =================================================================================================
class QueueFull(Exception):
    pass


class Job:
    def __init__(self, req):
        self.req = req
        self.out = queue.Queue()
        self.cancelled = threading.Event()
        self.t_enqueued = time.perf_counter()


class Scheduler:
    """Runs model.stream(prompt_ids, max_tokens, logprobs, info) for one job at a time, in arrival order.

    Items put on job.out: ("token", id, logprob_or_None, finish_reason_or_None, info_or_None) per token
    (finish_reason and info only on the last one), then ("end",); or ("error", message)."""

    def __init__(self, model, max_queue=0):
        self.model, self.max_queue = model, max_queue
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self.waiting = 0
        self.running = False
        self.completed = 0
        self._thread = threading.Thread(target=self._loop, name="engine-worker", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._q.put(None)
        self._thread.join(timeout=10)

    def submit(self, req):
        job = Job(req)
        with self._lock:
            if self.max_queue and self.waiting >= self.max_queue:
                raise QueueFull(f"{self.waiting} requests waiting (--max-queue {self.max_queue})")
            self.waiting += 1
        self._q.put(job)
        return job

    def status(self):
        with self._lock:
            return {"waiting": self.waiting, "running": self.running, "completed": self.completed}

    def _loop(self):
        while True:
            job = self._q.get()
            if job is None:
                return
            with self._lock:
                self.waiting -= 1
                self.running = True
            try:
                if not job.cancelled.is_set():
                    self._run(job)
            except Exception as e:  # noqa: BLE001 - e.g. a failed drain in gen.close(): keep serving
                log(f"[serve] worker error: {e!r}\n{traceback.format_exc()}")
                job.out.put(("error", f"engine error: {e!r}"))
            finally:
                with self._lock:
                    self.running = False
                    self.completed += 1

    def _run(self, job):
        req = job.req
        info = {"queue_ms": round((time.perf_counter() - job.t_enqueued) * 1e3, 3),
                "prompt_tokens": len(req.prompt_ids)}
        gen = self.model.stream(req.prompt_ids, req.max_tokens, req.logprobs, info)
        n = 0
        try:
            for tid, lp in gen:
                n += 1
                finish = "stop" if tid in req.stop_ids else ("length" if n >= req.max_tokens else None)
                job.out.put(("token", tid, lp, finish, info if finish else None))
                if finish or job.cancelled.is_set():
                    break
            else:
                if n < req.max_tokens:
                    raise RuntimeError(f"the model stopped after {n} of {req.max_tokens} tokens")
        except Exception as e:  # noqa: BLE001 - reported to the client, logged here
            log(f"[serve] engine error: {e!r}\n{traceback.format_exc()}")
            job.out.put(("error", f"engine error: {e!r}"))
            return
        finally:
            gen.close()   # drains in-flight GPU work before the next job
        job.out.put(("end",))


# =================================================================================================
# Response chunks
# =================================================================================================
def _logprob_entry_chat(tokenizer, tid, lp, top):
    s = tokenizer.decode([tid], skip_special_tokens=False)
    e = {"token": s, "logprob": lp, "bytes": list(s.encode("utf-8")), "top_logprobs": []}
    if top:
        e["top_logprobs"] = [{"token": s, "logprob": lp, "bytes": list(s.encode("utf-8"))}]
    return e


class ResponseBuilder:
    """Builds OpenAI chat / completion chunks and final bodies for one request."""

    def __init__(self, req, tokenizer, model_name):
        self.req, self.tok, self.model_name = req, tokenizer, model_name
        self.id = ("chatcmpl-" if req.kind == "chat" else "cmpl-") + uuid.uuid4().hex
        self.created = int(time.time())
        self.detok = IncrementalDetokenizer(tokenizer, req.skip_special_tokens)
        self.ids, self.lps, self.text, self.n = [], [], [], 0
        self.text_len = 0

    def _base(self, obj):
        return {"id": self.id, "object": obj, "created": self.created, "model": self.model_name}

    def token_chunk(self, tid, lp, finish):
        """One SSE chunk for one token (finish set on the last token)."""
        req = self.req
        self.n += 1
        delta = self.detok.add(tid)
        if finish:
            delta += self.detok.flush()
        self.ids.append(tid)
        self.lps.append(lp)
        self.text.append(delta)
        first = self.n == 1
        if req.kind == "chat":
            c = self._base("chat.completion.chunk")
            d = {"role": "assistant", "content": delta} if first else {"content": delta}
            ch = {"index": 0, "delta": d, "logprobs": None, "finish_reason": finish}
            if req.logprobs:
                ch["logprobs"] = {"content": [_logprob_entry_chat(self.tok, tid, lp, req.top_logprobs)]}
            if req.return_token_ids and first:
                c["prompt_token_ids"] = list(req.prompt_ids)
        else:
            c = self._base("text_completion")
            ch = {"index": 0, "text": delta, "logprobs": None, "finish_reason": finish}
            if req.logprobs:
                s = self.tok.decode([tid], skip_special_tokens=False)
                ch["logprobs"] = {"tokens": [s], "token_logprobs": [lp],
                                  "top_logprobs": [{s: lp}] if req.top_logprobs else None,
                                  "text_offset": [self.text_len]}
            if req.return_token_ids and first:
                ch["prompt_token_ids"] = list(req.prompt_ids)
        self.text_len += len(delta)
        if req.return_token_ids:
            ch["token_ids"] = [tid]
        c["choices"] = [ch]
        if req.include_usage:
            c["usage"] = None
        return c

    def usage(self):
        P = len(self.req.prompt_ids)
        return {"prompt_tokens": P, "completion_tokens": self.n, "total_tokens": P + self.n}

    def usage_chunk(self):
        c = self._base("chat.completion.chunk" if self.req.kind == "chat" else "text_completion")
        c["choices"] = []
        c["usage"] = self.usage()
        return c

    def final_body(self, finish, info):
        """The non-streaming response."""
        req = self.req
        text = "".join(self.text)
        if req.kind == "chat":
            body = self._base("chat.completion")
            ch = {"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None,
                  "finish_reason": finish}
            if req.logprobs:
                ch["logprobs"] = {"content": [_logprob_entry_chat(self.tok, t, lp, req.top_logprobs)
                                              for t, lp in zip(self.ids, self.lps)]}
            if req.return_token_ids:
                body["prompt_token_ids"] = list(req.prompt_ids)
        else:
            body = self._base("text_completion")
            ch = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish}
            if req.logprobs:
                toks = [self.tok.decode([t], skip_special_tokens=False) for t in self.ids]
                offs, o = [], 0
                for piece in self.text:
                    offs.append(o)
                    o += len(piece)
                ch["logprobs"] = {"tokens": toks, "token_logprobs": list(self.lps),
                                  "top_logprobs": [{s: lp} for s, lp in zip(toks, self.lps)] if req.top_logprobs else None,
                                  "text_offset": offs}
            if req.return_token_ids:
                ch["prompt_token_ids"] = list(req.prompt_ids)
        if req.return_token_ids:
            ch["token_ids"] = list(self.ids)
        body["choices"] = [ch]
        body["usage"] = self.usage()
        if info:
            body["inferencekit"] = info
        return body


# =================================================================================================
# HTTP
# =================================================================================================
class _ClientGone(Exception):
    pass


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen3-fp8-decode-serve"
    sys_version = ""

    def setup(self):
        super().setup()
        try:  # small per-token writes must not wait for Nagle / delayed ACK
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, fmt, *args):
        if self.server.access_log:
            log(f"[serve] {self.address_string()} {fmt % args}")

    # -- plumbing ----------------------------------------------------------------------------------
    def _send_json(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, err):
        self._send_json(err.status, err.body())

    def _chunk(self, data: bytes):
        try:
            self.wfile.write(b"%X\r\n%s\r\n" % (len(data), data))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as e:
            raise _ClientGone() from e

    def _event(self, obj):
        self._chunk(b"data: " + json.dumps(obj).encode() + b"\n\n")

    def _read_json(self):
        n = self.headers.get("Content-Length")
        if n is None:
            self.close_connection = True
            raise RequestError("Content-Length is required", status=411)
        try:
            n = int(n)
        except ValueError:
            self.close_connection = True
            raise RequestError("bad Content-Length")
        if n > MAX_BODY_BYTES:
            self.close_connection = True
            raise RequestError(f"body larger than {MAX_BODY_BYTES} bytes", status=413)
        raw = self.rfile.read(n)
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as e:
            raise RequestError(f"invalid JSON body: {e}")

    # -- routes ------------------------------------------------------------------------------------
    def do_GET(self):
        srv = self.server
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {"status": "ok", **srv.scheduler.status()})
        elif path == "/v1/models":
            self._send_json(200, {"object": "list", "data": [
                {"id": srv.served_model_name, "object": "model", "created": srv.started, "owned_by": ENGINE_NAME,
                 "root": srv.ctx.get("model_path"), "max_model_len": srv.ctx["max_seq"]}]})
        elif path == "/engine/config":
            self._send_json(200, srv.resolved_config)
        elif path in ("/v1/chat/completions", "/v1/completions", "/tokenize", "/detokenize"):
            self._error(RequestError(f"use POST for {path}", status=405))
        else:
            self._error(RequestError(f"not found: {path}", status=404, type_="NotFoundError"))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        routes = {"/v1/chat/completions": "chat", "/v1/completions": "completion"}
        try:
            if path not in routes and path not in ("/tokenize", "/detokenize"):
                if self.headers.get("Content-Length"):
                    self.rfile.read(int(self.headers["Content-Length"]))
                raise RequestError(f"not found: {path}", status=404, type_="NotFoundError")
            body = self._read_json()
            if path == "/tokenize":
                return self._tokenize(body)
            if path == "/detokenize":
                return self._detokenize(body)
            req = parse_request(routes[path], body, self.server.tokenizer, self.server.ctx)
        except RequestError as e:
            return self._error(e)
        except Exception as e:  # noqa: BLE001 - e.g. a chat template that raises on these messages
            return self._error(RequestError(f"could not build the prompt: {e!r}"))
        try:
            job = self.server.scheduler.submit(req)
        except QueueFull as e:
            return self._error(RequestError(f"server busy: {e}", status=503, type_="ServiceUnavailableError"))
        try:
            if req.stream:
                self._stream(job)
            else:
                self._complete(job)
        except _ClientGone:
            self.close_connection = True
        finally:
            job.cancelled.set()   # stops the worker if this handler left early; a no-op after the last token

    def _tokenize(self, body):
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        srv = self.server
        if "messages" in body:
            _check_fields(body, {"model", "messages", "add_generation_prompt", "chat_template_kwargs"})
            ids = chat_prompt_ids(body, srv.tokenizer)
        elif "prompt" in body:
            _check_fields(body, {"model", "prompt", "add_special_tokens"})
            if not isinstance(body["prompt"], str):
                raise RequestError("prompt must be a string", "prompt")
            ids = srv.tokenizer.encode(body["prompt"], add_special_tokens=_bool(body, "add_special_tokens", True))
        else:
            raise RequestError("send prompt or messages")
        self._send_json(200, {"tokens": ids, "count": len(ids), "max_model_len": srv.ctx["max_seq"]})

    def _detokenize(self, body):
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        _check_fields(body, {"model", "tokens"})
        toks = body.get("tokens")
        if not (isinstance(toks, list) and all(_is_int(t) for t in toks)):
            raise RequestError("tokens must be a list of integers", "tokens")
        self._send_json(200, {"prompt": self.server.tokenizer.decode(toks, skip_special_tokens=False)})

    def _stream(self, job):
        req = job.req
        rb = ResponseBuilder(req, self.server.tokenizer, self.server.served_model_name)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:
            item = job.out.get()
            if item[0] == "token":
                _, tid, lp, finish, info = item
                c = rb.token_chunk(tid, lp, finish)
                if finish and not req.include_usage and info:
                    c["inferencekit"] = info
                self._event(c)
                if finish:
                    if req.include_usage:
                        u = rb.usage_chunk()
                        if info:
                            u["inferencekit"] = info
                        self._event(u)
                    break
            elif item[0] == "error":
                self._event(RequestError(item[1], status=500, type_="InternalServerError").body())
                break
            else:   # "end" without a finishing token (cancelled)
                break
        self._chunk(b"data: [DONE]\n\n")
        self._chunk(b"")   # zero-length chunk: end of the chunked body

    def _complete(self, job):
        req = job.req
        rb = ResponseBuilder(req, self.server.tokenizer, self.server.served_model_name)
        finish, info = None, None
        while True:
            item = job.out.get()
            if item[0] == "token":
                _, tid, lp, finish, info = item
                rb.token_chunk(tid, lp, finish)
                if finish:
                    break
            elif item[0] == "error":
                return self._error(RequestError(item[1], status=500, type_="InternalServerError"))
            else:
                break
        self._send_json(200, rb.final_body(finish, info))


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024   # listen backlog: load tests open many connections at once

    def handle_error(self, request, client_address):
        # a client closing its connection (e.g. after reading what it needed) is not a server error
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


def make_server(model, tokenizer, *, host="127.0.0.1", port=0, served_model_name=ENGINE_NAME, model_path=None,
                eos_ids=(), vocab_size=None, resolved_config=None, max_queue=0, access_log=False):
    """Build (not start) the HTTP server and its scheduler. Tests inject a fake model and tokenizer."""
    srv = Server((host, port), Handler)
    srv.tokenizer = tokenizer
    srv.served_model_name = served_model_name
    srv.started = int(time.time())
    srv.access_log = access_log
    names = {served_model_name}
    if model_path:
        names.add(model_path)
    srv.ctx = {"model_names": names, "model_path": model_path, "max_seq": model.max_seq, "eos_ids": tuple(eos_ids),
               "vocab_size": vocab_size, "logprobs_supported": bool(getattr(model, "logprobs_supported", False))}
    srv.resolved_config = resolved_config or {}
    srv.scheduler = Scheduler(model, max_queue=max_queue).start()
    return srv


def format_config_line(cfg: dict) -> str:
    return CONFIG_PREFIX + json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def format_shared_config_line(cfg: dict) -> str:
    """The same config in the line format that inferencekit's server runner reads from any custom engine's
    log (`{"inferencekit_resolved_config": {...}}`), so a managed run records it as resolved."""
    return json.dumps({"inferencekit_resolved_config": cfg}, sort_keys=True, separators=(",", ":"))


def parse_config_line(line: str):
    """The resolved-config JSON from a log line, or None if the line is not one."""
    i = line.find(CONFIG_PREFIX)
    return json.loads(line[i + len(CONFIG_PREFIX):]) if i >= 0 else None


# =================================================================================================
# The real engine (torch imported lazily so the HTTP layer is testable without a GPU)
# =================================================================================================
def pipelined(n, lookahead, launch, collect):
    """Yield collect(0..n-1) in order while keeping up to `lookahead` launched-but-uncollected steps.
    launch(i) enqueues step i (non-blocking); collect(i) blocks until step i's result is on the host.
    Step i + lookahead is launched before step i's result is yielded, so the device keeps working
    while the consumer handles the result. Closing the generator stops further launches."""
    launched = 0
    while launched < min(n, max(1, lookahead)):
        launch(launched)
        launched += 1
    for i in range(n):
        r = collect(i)
        if launched < n:
            launch(launched)
            launched += 1
        yield r


class HFTokenizer:
    """transformers AutoTokenizer: encode/decode and the model's own chat template (as the Hugging Face
    reference path). Calls are serialized: fast tokenizers are not guaranteed
    thread-safe and requests are tokenized in their handler threads."""

    def __init__(self, path):
        # This directory has a package named `kernels`, which would shadow the Hub `kernels` package that
        # transformers may probe for (same workaround as check_correctness.py hf).
        saved = list(sys.path)
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _HERE]
        try:
            from transformers import AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path)
            import transformers
            self.version = transformers.__version__
        finally:
            sys.path[:] = saved
        self._lock = threading.Lock()

    def encode(self, text, add_special_tokens=True):
        with self._lock:
            return list(self.tok.encode(text, add_special_tokens=add_special_tokens))

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        with self._lock:
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                                **kwargs)
            # The template already holds the special tokens (vLLM's chat path does the same).
            return list(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids, skip_special_tokens=True):
        with self._lock:
            return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


def model_eos_ids(path, tokenizer):
    """EOS ids from generation_config.json (Qwen3: <|im_end|>, <|endoftext|>) plus the tokenizer's."""
    ids = set()
    try:
        with open(os.path.join(path, "generation_config.json")) as f:
            e = json.load(f).get("eos_token_id")
        ids.update(e if isinstance(e, list) else [e] if e is not None else [])
    except (OSError, ValueError):
        pass
    if getattr(tokenizer.tok, "eos_token_id", None) is not None:
        ids.add(tokenizer.tok.eos_token_id)
    return sorted(ids)


class EngineModel:
    """generate.Engine as a token stream. Loading and warm-up follow bench.py."""

    def __init__(self, args):
        import torch
        from generate import Engine
        t = time.perf_counter()
        eng = Engine(args.model, max_seq=args.max_seq, pdl=not args.no_pdl, prefill_copy=not args.no_prefill_copy,
                     prefill_graphs=not args.no_prefill_graphs)
        self.load_s = time.perf_counter() - t
        self._setup(torch, eng, args.lookahead, args.logprobs)
        # distinct lengths only: a repeated length would be its 2nd sighting and capture a prefill graph
        self.warmup_lengths = [n for n in dict.fromkeys(args.warmup_lengths) if 0 < n and n + 2 <= self.max_seq]
        t = time.perf_counter()
        with torch.no_grad():
            for n in self.warmup_lengths:
                # bench.py's untimed first call: eager prefill (compiles the Triton kernels for this tile
                # class) + decode-graph capture on the first one.
                eng.generate([(i * 7919 + 13) % 150000 + 100 for i in range(n)], 2)
        torch.cuda.synchronize()
        self.warmup_s = time.perf_counter() - t
        # Forget the warm-up sightings: live traffic decides which prompt lengths get a prefill graph.
        eng._prefill_seen.clear()
        assert eng.graph is not None, "decode graph not captured"
        assert not eng._prefill_graphs

    def _setup(self, torch, eng, lookahead, logprobs, device="cuda"):
        """Streaming buffers and the optional prefill logprob hook (tests call this with a fake torch)."""
        self.torch, self.engine = torch, eng
        self.max_seq = eng.max_seq
        self.vocab_size = eng.m.cfg.vocab
        self.lookahead = max(1, int(lookahead))
        self.logprobs_supported = bool(logprobs)
        self.copy_stream = torch.cuda.Stream()
        self.host_tok = torch.zeros(self.max_seq, dtype=torch.int32, pin_memory=True)
        self.host_lp = torch.zeros(self.max_seq, dtype=torch.float32, pin_memory=True)
        self.lp_dev = torch.zeros(self.max_seq, dtype=torch.float32, device=device)
        self._impl_by_len = {}
        ring = self.lookahead + 2   # > lookahead: an event is re-recorded only after its step was collected
        self.ev_step = [torch.cuda.Event() for _ in range(ring)]
        self.copied_events = [torch.cuda.Event() for _ in range(ring)]
        if self.logprobs_supported:
            # The first token comes from prefill, whose logits stay inside Engine._prefill_body. Wrap the
            # model's forward so every prefill (also inside its CUDA graph) writes the chosen token's
            # logprob to a static buffer. Installed before any prefill graph is captured.
            self.first_lp = torch.zeros(1, dtype=torch.float32, device=device)
            orig, buf = eng.m.forward, self.first_lp

            def forward_with_logprob(tokens, start):
                logits = orig(tokens, start)
                lf = logits.float()
                buf.copy_((lf.max() - torch.logsumexp(lf, 0)).view(1))
                return logits

            eng.m.forward = forward_with_logprob

    def resolved_config(self, args, tokenizer, eos_ids, served_model_name):
        import bench
        from generate import source_identity
        eng, m = self.engine, self.engine.m
        layer0 = m.layers[0]
        return {
            "engine": ENGINE_NAME, "served_model_name": served_model_name, "model_path": args.model,
            "source": source_identity(), "versions": {**bench.versions(), "transformers": tokenizer.version},
            # top-level key read by inferencekit's CUDA-graph gate for custom engines: the decode graph exists
            "cuda_graphs": eng.graph is not None,
            "settings": eng.settings(),
            "observed": {
                "decode_graph_captured": eng.graph is not None,
                "prefill_graphs": eng.prefill_graphs,
                "prefill_graph_lengths": sorted(eng._prefill_graphs),
                "kv_cache_dtype": str(m.k_cache.dtype).replace("torch.", ""),
                "kv_cache_shape": list(m.k_cache.shape),
                "weight_dtype": str(layer0.w_qkv.dtype).replace("torch.", ""),
                "lm_head_dtype": str(m.lm_head.dtype).replace("torch.", ""),
                "max_context": self.max_seq,
                "vocab_size": self.vocab_size,
                "load_s": round(self.load_s, 2), "warmup_s": round(self.warmup_s, 2),
            },
            "kernels": {
                "decode": "kernels/decode.cu in one CUDA graph per step: embed, 36 x [fp8_gemv qkv, fused attention "
                          "(q/k norm, RoPE, KV write, split-K), fp8_gemv o_proj, fp8_gemv gate|up, fp8_gemv down], "
                          "lm_head_argmax, finalize",
                "prefill": "Triton FP8 block matmul (kernels/prefill.py) + fused elementwise Triton kernels "
                           "(kernels/prefill_fused.py) for >= 16 tokens, PyTorch ops below; attention: torch SDPA",
                "prefill_impl": m.prefill_impl,
            },
            "server": {
                "batch": 1, "scheduling": "FIFO, one request at a time", "max_queue": args.max_queue,
                "lookahead": self.lookahead, "logprobs": self.logprobs_supported,
                "sampling": "greedy (temperature 0) only", "warmup_lengths": self.warmup_lengths,
                "eos_token_ids": list(eos_ids), "chat_template": "the tokenizer's own (transformers apply_chat_template)",
                "host": args.host, "port": args.port,
            },
        }

    def stream(self, prompt_ids, max_tokens, logprobs, info):
        torch, eng = self.torch, self.engine
        torch.set_grad_enabled(False)   # thread-local: this runs on the worker thread
        P = len(prompt_ids)
        assert P + max_tokens <= self.max_seq
        R = len(self.ev_step)
        graphs_before = set(eng._prefill_graphs)
        first = eng.prefill(prompt_ids)          # device tensor; no host sync
        info["prefill"] = ("graph_replay" if P in graphs_before else
                           "graph_capture" if P in eng._prefill_graphs else "eager")
        # a replay runs no Python, so last_prefill_path is only fresh for eager runs and captures
        if info["prefill"] != "graph_replay":
            self._impl_by_len[P] = eng.m.last_prefill_path
        info["prefill_impl"] = self._impl_by_len.get(P)
        want_lp = logprobs and self.logprobs_supported

        def launch(s):
            # step 0 = the prefill's token; step s >= 1 = decode-graph replay s - 1, which writes out_tokens[s - 1]
            if s == 0:
                src, lp_src = first.view(1), (self.first_lp if want_lp else None)
            else:
                eng.graph.replay()
                src = eng.out_tokens[s - 1:s]
                lp_src = None
                if want_lp:
                    lf = eng.logits.float()   # this step's logits; the argmax is the chosen token
                    self.lp_dev[s:s + 1].copy_((lf.max() - torch.logsumexp(lf, 0)).view(1))
                    lp_src = self.lp_dev[s:s + 1]
            ev = self.ev_step[s % R]
            ev.record()
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(ev)
                self.host_tok[s:s + 1].copy_(src, non_blocking=True)
                if lp_src is not None:
                    self.host_lp[s:s + 1].copy_(lp_src, non_blocking=True)
                self.copied_events[s % R].record(self.copy_stream)

        def collect(s):
            self.copied_events[s % R].synchronize()
            return int(self.host_tok[s]), (float(self.host_lp[s]) if want_lp else None)

        try:
            yield from pipelined(max_tokens, self.lookahead, launch, collect)
        finally:
            torch.cuda.synchronize()   # in-flight replays (after a stop token) finish before the next job
            del first


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="local checkpoint directory (as bench.py --model)")
    ap.add_argument("--tokenizer", help="tokenizer path (default: --model)")
    ap.add_argument("--served-model-name", help="model name in the API (default: --model as given)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-seq", type=int, default=4096, help="max context (prompt + generated), as bench.py")
    ap.add_argument("--no-pdl", action="store_true", help="disable programmatic dependent launch (as bench.py)")
    ap.add_argument("--no-prefill-copy", action="store_true", help="low-VRAM prefill (as bench.py)")
    ap.add_argument("--no-prefill-graphs", action="store_true",
                    help="never CUDA-graph the prefill: every TTFT is the eager one (as bench.py)")
    ap.add_argument("--logprobs", action="store_true",
                    help="allow logprobs (adds a vocabulary log-softmax to every prefill)")
    ap.add_argument("--lookahead", type=int, default=4, help="decode steps queued ahead of the host (>= 1)")
    ap.add_argument("--max-queue", type=int, default=0, help="503 beyond this many waiting requests (0: no limit)")
    ap.add_argument("--warmup-lengths", default=",".join(map(str, DEFAULT_WARMUP_LENGTHS)),
                    help="comma-separated prompt lengths run once at startup (Triton compilation)")
    ap.add_argument("--access-log", action="store_true")
    args = ap.parse_args(argv)
    try:
        args.warmup_lengths = [int(x) for x in args.warmup_lengths.split(",") if x.strip()]
    except ValueError:
        ap.error("--warmup-lengths must be comma-separated integers")
    if not args.warmup_lengths:
        ap.error("--warmup-lengths needs at least one length (the first one captures the decode graph)")
    served = args.served_model_name or args.model

    t = time.perf_counter()
    tokenizer = HFTokenizer(args.tokenizer or args.model)   # before the engine import (sys.path workaround)
    eos_ids = model_eos_ids(args.model, tokenizer)
    log(f"[serve] tokenizer loaded in {time.perf_counter() - t:.1f}s; eos ids {eos_ids}")
    sys.path.insert(0, _HERE)
    model = EngineModel(args)
    cfg = model.resolved_config(args, tokenizer, eos_ids, served)
    log(format_config_line(cfg))
    log(format_shared_config_line(cfg))
    srv = make_server(model, tokenizer, host=args.host, port=args.port, served_model_name=served,
                      model_path=args.model, eos_ids=eos_ids, vocab_size=model.vocab_size, resolved_config=cfg,
                      max_queue=args.max_queue, access_log=args.access_log)
    log(f"[serve] listening on http://{args.host}:{srv.server_address[1]} (model {served!r}, batch 1)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        srv.scheduler.stop()


if __name__ == "__main__":
    main()
