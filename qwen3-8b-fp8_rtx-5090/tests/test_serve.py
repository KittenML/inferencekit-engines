"""HTTP server tests with a fake model and a fake tokenizer (no GPU, no torch, no transformers).

Run: uv run --all-extras pytest -q engines/qwen3-fp8-decode/tests
"""
from __future__ import annotations

import http.client
import importlib.util
import json
import socket
import threading
import time
from pathlib import Path

import pytest

ENGINE_DIR = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("qwen3_fp8_serve", ENGINE_DIR / "serve.py")
serve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(serve)

EOS = 2
SPECIAL = {0: "<bos>", 1: "<im>", EOS: "<eos>"}


class FakeTokenizer:
    """Character-level: id = Unicode code point; ids 0-2 are special tokens."""

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        text = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        if kwargs.get("enable_thinking") is False:
            text += "~"
        return self.encode(text)

    def decode(self, ids, skip_special_tokens=True):
        return "".join(("" if skip_special_tokens else SPECIAL[i]) if i in SPECIAL else chr(i) for i in ids)


class ByteTokenizer:
    """Byte-level: id = one UTF-8 byte, so a character can span several tokens."""

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="replace")


def ids(text):
    return [ord(c) for c in text]


class FakeModel:
    """Streams a fixed token list (cycled), one token per `delay` seconds, and records every call."""

    def __init__(self, tokens, delay=0.0, max_seq=64, logprobs_supported=True):
        self.tokens, self.delay, self.max_seq = list(tokens), delay, max_seq
        self.logprobs_supported = logprobs_supported
        self.calls = []
        self._lock = threading.Lock()

    def stream(self, prompt_ids, max_tokens, logprobs, info):
        call = {"prompt_ids": list(prompt_ids), "max_tokens": max_tokens, "logprobs": logprobs,
                "t_start": time.perf_counter(), "t_end": None, "yielded": 0, "closed_early": False}
        with self._lock:
            self.calls.append(call)
        info["prefill"] = "fake"
        try:
            for i in range(max_tokens):
                if self.delay:
                    time.sleep(self.delay)
                call["yielded"] += 1
                yield self.tokens[i % len(self.tokens)], (-0.25 * (i + 1) if logprobs else None)
        except GeneratorExit:
            call["closed_early"] = call["yielded"] < max_tokens
            raise
        finally:
            call["t_end"] = time.perf_counter()


TOKENS = ids("Hello") + [EOS] + ids(" world")   # EOS at index 5


@pytest.fixture
def server():
    started = []

    def start(model=None, tokenizer=None, **kw):
        model = model or FakeModel(TOKENS)
        srv = serve.make_server(model, tokenizer or FakeTokenizer(), host="127.0.0.1", port=0,
                                served_model_name="fake-model", eos_ids=(EOS,), vocab_size=1 << 20,
                                resolved_config={"engine": "fake"}, **kw)
        th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
        th.start()
        started.append(srv)
        return srv.server_address[1], model

    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()
        srv.scheduler.stop()


# --------------------------------------------------------------------------------------------------
# client helpers
# --------------------------------------------------------------------------------------------------
def request(port, method, path, body=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, path, body=data, headers={"Content-Type": "application/json"} if data else {})
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        return r.status, dict(r.getheaders()), json.loads(raw)
    except ValueError:
        return r.status, dict(r.getheaders()), raw


def sse(port, path, body, on_event=None, stop_after=None):
    """POST and read the SSE stream. Returns (status, headers, [(t_arrival, event_json_or_DONE)])."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", path, body=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = c.getresponse()
    headers = dict(r.getheaders())
    if r.status != 200:
        out = json.loads(r.read())
        c.close()
        return r.status, headers, out
    events = []
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        assert line.startswith(b"data: "), line
        payload = line[len(b"data: "):]
        ev = "[DONE]" if payload == b"[DONE]" else json.loads(payload)
        events.append((time.perf_counter(), ev))
        if on_event:
            on_event(ev)
        if ev == "[DONE]" or (stop_after and len(events) >= stop_after):
            break
    if stop_after and events[-1][1] != "[DONE]":
        c.sock.shutdown(socket.SHUT_RDWR)
    c.close()
    return 200, headers, events


def chat(**kw):
    body = {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "temperature": 0,
            "stream": True}
    body.update(kw)
    return body


def completion(**kw):
    body = {"model": "fake-model", "prompt": "hi", "temperature": 0, "stream": True}
    body.update(kw)
    return body


# --------------------------------------------------------------------------------------------------
# streaming format
# --------------------------------------------------------------------------------------------------
def test_chat_stream_sse_format_and_usage(server):
    port, model = server()
    status, headers, events = sse(port, "/v1/chat/completions",
                                  chat(max_tokens=20, stream_options={"include_usage": True}))
    assert status == 200
    assert headers["Content-Type"] == "text/event-stream"
    assert events[-1][1] == "[DONE]"
    chunks = [e for _, e in events[:-1]]
    token_chunks, usage_chunk = chunks[:-1], chunks[-1]
    # one chunk per generated token, up to and including EOS
    assert len(token_chunks) == 6
    assert [c["choices"][0]["token_ids"] for c in token_chunks] == [[t] for t in TOKENS[:6]]
    ids_ = {c["id"] for c in chunks}
    assert len(ids_) == 1 and next(iter(ids_)).startswith("chatcmpl-")
    for c in token_chunks:
        assert c["object"] == "chat.completion.chunk" and c["model"] == "fake-model"
        assert c["usage"] is None and len(c["choices"]) == 1 and c["choices"][0]["index"] == 0
    first = token_chunks[0]
    assert first["choices"][0]["delta"] == {"role": "assistant", "content": "H"}
    assert first["prompt_token_ids"] == FakeTokenizer().apply_chat_template([{"role": "user", "content": "hi"}])
    assert all("role" not in c["choices"][0]["delta"] for c in token_chunks[1:])
    assert "".join(c["choices"][0]["delta"]["content"] for c in token_chunks) == "Hello"   # EOS text skipped
    assert [c["choices"][0]["finish_reason"] for c in token_chunks] == [None] * 5 + ["stop"]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"] == {"prompt_tokens": len(first["prompt_token_ids"]), "completion_tokens": 6,
                                    "total_tokens": len(first["prompt_token_ids"]) + 6}
    assert usage_chunk["inferencekit"]["prefill"] == "fake"
    assert usage_chunk["inferencekit"]["prompt_tokens"] == len(first["prompt_token_ids"])
    assert model.calls[0]["max_tokens"] == 20 and model.calls[0]["closed_early"]


def test_stream_without_usage_has_no_usage_fields(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=3))
    chunks = [e for _, e in events[:-1]]
    assert len(chunks) == 3 and all("usage" not in c for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert "inferencekit" in chunks[-1]


def test_ignore_eos_generates_exactly_max_tokens(server):
    port, model = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=15, ignore_eos=True, stream_options={"include_usage": True}))
    chunks = [e for _, e in events[:-1]]
    toks = [c["choices"][0]["token_ids"][0] for c in chunks[:-1]]
    assert toks == (TOKENS * 2)[:15] and EOS in toks
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 15
    assert not model.calls[0]["closed_early"]


def test_max_tokens_before_eos_finishes_with_length(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_completion_tokens=3))
    chunks = [e for _, e in events[:-1]]
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks) == "Hel"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"


def test_stop_token_ids_apply_even_with_ignore_eos(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=30, ignore_eos=True, stop_token_ids=[ord("w")]))
    chunks = [e for _, e in events[:-1]]
    text = "".join(c["choices"][0]["delta"]["content"] for c in chunks)
    assert text == "Hello w" and chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_return_token_ids_false_omits_extension(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=2, return_token_ids=False))
    chunks = [e for _, e in events[:-1]]
    assert all("token_ids" not in c["choices"][0] and "prompt_token_ids" not in c for c in chunks)


def test_completions_stream_with_token_id_prompt_and_logprobs(server):
    port, model = server()
    prompt = [72, 105, 33]
    _, _, events = sse(port, "/v1/completions", completion(prompt=prompt, max_tokens=4, logprobs=1))
    chunks = [e for _, e in events[:-1]]
    assert len(chunks) == 4
    assert model.calls[0]["prompt_ids"] == prompt and model.calls[0]["logprobs"] is True
    c0 = chunks[0]
    assert c0["object"] == "text_completion" and c0["id"].startswith("cmpl-")
    assert c0["choices"][0]["prompt_token_ids"] == prompt
    assert [c["choices"][0]["text"] for c in chunks] == ["H", "e", "l", "l"]
    lp = [c["choices"][0]["logprobs"] for c in chunks]
    assert [x["token_logprobs"] for x in lp] == [[-0.25], [-0.5], [-0.75], [-1.0]]
    assert [x["tokens"] for x in lp] == [["H"], ["e"], ["l"], ["l"]]
    assert [x["text_offset"] for x in lp] == [[0], [1], [2], [3]]
    assert lp[0]["top_logprobs"] == [{"H": -0.25}]


def test_chat_logprobs_format(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=2, logprobs=True))
    c = events[0][1]["choices"][0]["logprobs"]["content"][0]
    assert c == {"token": "H", "logprob": -0.25, "bytes": [72], "top_logprobs": []}


def test_chat_template_kwargs_reach_the_template(server):
    port, _ = server()
    _, _, events = sse(port, "/v1/chat/completions",
                       chat(max_tokens=1, chat_template_kwargs={"enable_thinking": False}))
    assert events[0][1]["prompt_token_ids"][-1] == ord("~")


def test_non_streaming_chat_and_completion(server):
    port, _ = server()
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=20))
    assert status == 200 and body["object"] == "chat.completion"
    ch = body["choices"][0]
    assert ch["message"] == {"role": "assistant", "content": "Hello"} and ch["finish_reason"] == "stop"
    assert ch["token_ids"] == TOKENS[:6] and body["usage"]["completion_tokens"] == 6
    status, _, body = request(port, "POST", "/v1/completions", completion(stream=False, max_tokens=8, logprobs=0))
    ch = body["choices"][0]
    assert status == 200 and ch["text"] == "Hello" and ch["finish_reason"] == "stop"
    assert ch["logprobs"]["token_logprobs"] == [-0.25 * (i + 1) for i in range(6)]
    assert ch["logprobs"]["top_logprobs"] is None


# --------------------------------------------------------------------------------------------------
# rejection: unsupported parameters are 400, never silently ignored
# --------------------------------------------------------------------------------------------------
REJECT = [
    ("chat", {"temperature": None}, "temperature"),
    ("chat", {"temperature": 0.7}, "temperature"),
    ("chat", {"temperature": True}, "temperature"),
    ("completion", {"temperature": 1}, "temperature"),
    ("chat", {"n": 2}, "n"),
    ("chat", {"stop": ["\n"]}, "stop"),
    ("chat", {"presence_penalty": 0.5}, "presence_penalty"),
    ("chat", {"frequency_penalty": 0.1}, "frequency_penalty"),
    ("chat", {"repetition_penalty": 1.1}, "repetition_penalty"),
    ("chat", {"logit_bias": {"5": 10}}, "logit_bias"),
    ("chat", {"min_tokens": 4}, "min_tokens"),                                   # would mask EOS
    ("chat", {"min_tokens": 1}, "min_tokens"),
    ("chat", {"min_tokens": 4, "ignore_eos": True, "stop_token_ids": [7]}, "min_tokens"),  # would mask id 7
    ("chat", {"min_tokens": 5, "ignore_eos": True}, "min_tokens"),               # > max_tokens (4)
    ("chat", {"min_tokens": "4", "ignore_eos": True}, "min_tokens"),
    ("chat", {"min_tokens": -1, "ignore_eos": True}, "min_tokens"),
    ("completion", {"min_tokens": 4}, "min_tokens"),
    ("chat", {"top_logprobs": 5, "logprobs": True}, "top_logprobs"),
    ("chat", {"top_logprobs": 1}, "top_logprobs"),
    ("chat", {"response_format": {"type": "json_object"}}, "response_format"),
    ("chat", {"tools": []}, "tools"),
    ("chat", {"max_tokens": 0}, "max_tokens"),
    ("chat", {"max_tokens": 1000}, "max_tokens"),           # prompt + 1000 > max_seq 64
    ("chat", {"max_tokens": 5, "max_completion_tokens": 6}, "max_tokens"),
    ("chat", {"stream": False, "stream_options": {"include_usage": True}}, "stream_options"),
    ("chat", {"stream_options": {"continuous_usage_stats": True}}, "stream_options"),
    ("chat", {"messages": []}, "messages"),
    ("chat", {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]}, "messages"),
    ("chat", {"ignore_eos": "yes"}, "ignore_eos"),
    ("completion", {"echo": True}, "echo"),
    ("completion", {"best_of": 3}, "best_of"),
    ("completion", {"suffix": "x"}, "suffix"),
    ("completion", {"logprobs": 5}, "logprobs"),
    ("completion", {"prompt": ["a", "b"]}, "prompt"),
    ("completion", {"prompt": ""}, "prompt"),
    ("completion", {"prompt": [1, -5]}, "prompt"),
]


@pytest.mark.parametrize("kind,patch,param", REJECT, ids=[f"{k}-{p}-{i}" for i, (k, _, p) in enumerate(REJECT)])
def test_unsupported_parameters_are_rejected(server, kind, patch, param):
    port, model = server()
    body = chat(max_tokens=4) if kind == "chat" else completion(max_tokens=4)
    for k, v in patch.items():
        if v is None:
            body.pop(k, None)
        else:
            body[k] = v
    path = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
    status, _, out = request(port, "POST", path, body)
    assert status == 400, out
    assert out["error"]["param"] == param and out["error"]["message"]
    assert model.calls == []


def test_accepted_noop_sampling_params(server):
    port, model = server()
    status, _, body = request(port, "POST", "/v1/chat/completions",
                              chat(stream=False, max_tokens=2, top_p=0.9, top_k=20, min_p=0.0, seed=7,
                                   presence_penalty=0, repetition_penalty=1, n=1, stop=[], user="u"))
    assert status == 200 and body["choices"][0]["message"]["content"] == "He"


# The bodies `inferencekit bench serve` builds per --engine-kind (serve/profiles.py build_body
# with the default token_ids=auto and exact_length=true), written out literally: the plugin package is
# not imported here. vllm/sglang/trtllm force the length with ignore_eos + min_tokens = max_tokens.
BENCH_SERVE_EXACT = {
    "vllm": {"ignore_eos": True, "min_tokens": "{n}"},
    "sglang": {"ignore_eos": True, "min_tokens": "{n}"},
    "trtllm": {"ignore_eos": True, "min_tokens": "{n}"},
    "llamacpp": {"ignore_eos": True},
}
BENCH_SERVE_IDS = {"vllm": {"return_token_ids": True}}


def bench_serve_body(api, kind, n):
    body = {"model": "fake-model"}
    msgs = [{"role": "user", "content": "hi"}]
    if api == "chat":
        body["messages"] = msgs
    else:
        body["prompt"] = "\n\n".join(m["content"] for m in msgs)
    body.update({"max_tokens": n, "temperature": 0.0, "stream": True, "stream_options": {"include_usage": True}})
    body.update({k: (n if v == "{n}" else v) for k, v in BENCH_SERVE_EXACT.get(kind, {}).items()})
    body.update(BENCH_SERVE_IDS.get(kind, {}))
    return body


@pytest.mark.parametrize("api", ["chat", "completions"])
@pytest.mark.parametrize("kind", ["vllm", "sglang", "trtllm", "llamacpp"])
def test_bench_serve_exact_length_bodies_get_exactly_max_tokens(server, api, kind):
    """ignore_eos + min_tokens <= max_tokens with no stop_token_ids masks nothing, so it is accepted."""
    port, model = server()
    n = 20
    path = "/v1/chat/completions" if api == "chat" else "/v1/completions"
    status, _, events = sse(port, path, bench_serve_body(api, kind, n))
    assert status == 200, events
    chunks = [e for _, e in events[:-1]]
    token_chunks = [c for c in chunks if c["choices"]]
    assert [c["choices"][0]["token_ids"][0] for c in token_chunks] == (TOKENS * 3)[:n]   # EOS (id 2) included
    assert token_chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] == n
    assert model.calls[0]["max_tokens"] == n and not model.calls[0]["closed_early"]


def test_min_tokens_zero_is_always_a_noop(server):
    port, _ = server()
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=20, min_tokens=0))
    assert status == 200 and body["choices"][0]["finish_reason"] == "stop"


def test_logprobs_need_server_support(server):
    port, model = server(FakeModel(TOKENS, logprobs_supported=False))
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(logprobs=True, max_tokens=2))
    assert status == 400 and "--logprobs" in out["error"]["message"] and model.calls == []


def test_unknown_model_is_404_and_bad_json_is_400(server):
    port, _ = server()
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(model="other"))
    assert status == 404 and out["error"]["code"] == "model_not_found"
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/chat/completions", body=b"{not json", headers={"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 400 and "invalid JSON" in json.loads(r.read())["error"]["message"]
    status, _, _ = request(port, "POST", "/v1/nothing", {})
    assert status == 404
    status, _, _ = request(port, "GET", "/v1/chat/completions")
    assert status == 405


# --------------------------------------------------------------------------------------------------
# batch 1: queueing
# --------------------------------------------------------------------------------------------------
def test_requests_queue_fifo_one_at_a_time(server):
    port, model = server(FakeModel(TOKENS, delay=0.02))
    results, health = {}, []

    def client(i):
        body = chat(max_tokens=5, messages=[{"role": "user", "content": f"q{i}"}],
                    stream_options={"include_usage": True})
        results[i] = sse(port, "/v1/chat/completions", body)

    threads = []
    for i in range(3):
        t = threading.Thread(target=client, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.03)   # fixes the arrival order
    health.append(request(port, "GET", "/health")[2])
    for t in threads:
        t.join(10)
    # the model ran the three requests one after another, in arrival order
    calls = model.calls
    assert len(calls) == 3
    assert [c["prompt_ids"][-len("q0[assistant]"):] for c in calls] == [ids(f"q{i}[assistant]") for i in range(3)]
    for a, b in zip(calls, calls[1:]):
        assert a["t_end"] <= b["t_start"]
    # client side: each stream's first token arrives after the previous stream's last token
    spans = []
    for i in range(3):
        status, _, events = results[i]
        assert status == 200
        tok_times = [t for t, e in events if e != "[DONE]" and e["choices"]]
        assert len(tok_times) == 5
        spans.append((tok_times[0], tok_times[-1]))
        usage = [e for _, e in events if e != "[DONE]" and not e["choices"]][0]
        if i > 0:
            assert usage["inferencekit"]["queue_ms"] > 20   # it waited for the request ahead of it
    for (_, last), (first, _) in zip(spans, spans[1:]):
        assert last <= first
    assert health[0]["status"] == "ok" and health[0]["running"] and health[0]["waiting"] >= 1


def test_max_queue_answers_503(server):
    port, model = server(FakeModel(TOKENS, delay=0.05), max_queue=1)
    done = []
    t1 = threading.Thread(target=lambda: done.append(sse(port, "/v1/chat/completions", chat(max_tokens=4))))
    t1.start()
    time.sleep(0.05)   # request 1 is running
    t2 = threading.Thread(target=lambda: done.append(sse(port, "/v1/chat/completions", chat(max_tokens=4))))
    t2.start()
    time.sleep(0.05)   # request 2 is waiting
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(max_tokens=4))
    assert status == 503 and "busy" in out["error"]["message"]
    t1.join(10)
    t2.join(10)
    assert [d[0] for d in done] == [200, 200]


def test_client_disconnect_stops_generation_and_frees_the_engine(server):
    port, model = server(FakeModel(ids("abcdefghij"), delay=0.01, max_seq=256))
    status, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=60), stop_after=2)
    assert status == 200 and len(events) == 2
    deadline = time.time() + 5
    while time.time() < deadline and (not model.calls or model.calls[0]["t_end"] is None):
        time.sleep(0.01)
    assert model.calls[0]["closed_early"] and model.calls[0]["yielded"] < 60
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=3))
    assert status == 200 and body["choices"][0]["message"]["content"] == "abc"


def test_engine_error_mid_stream_is_reported(server):
    class Broken(FakeModel):
        def stream(self, prompt_ids, max_tokens, logprobs, info):
            yield 72, None
            raise RuntimeError("boom")

    port, _ = server(Broken(TOKENS))
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=5))
    assert events[0][1]["choices"][0]["delta"]["content"] == "H"
    assert "boom" in events[1][1]["error"]["message"] and events[-1][1] == "[DONE]"
    status, _, out = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=5))
    assert status == 500 and "boom" in out["error"]["message"]


# --------------------------------------------------------------------------------------------------
# other endpoints
# --------------------------------------------------------------------------------------------------
def test_health_models_config_tokenize(server):
    port, _ = server()
    status, _, h = request(port, "GET", "/health")
    assert status == 200 and h["status"] == "ok" and h["waiting"] == 0
    status, _, m = request(port, "GET", "/v1/models")
    assert m["data"][0]["id"] == "fake-model" and m["data"][0]["max_model_len"] == 64
    assert request(port, "GET", "/engine/config")[2] == {"engine": "fake"}
    status, _, t = request(port, "POST", "/tokenize", {"prompt": "hey"})
    assert status == 200 and t == {"tokens": ids("hey"), "count": 3, "max_model_len": 64}
    status, _, t = request(port, "POST", "/tokenize", {"messages": [{"role": "user", "content": "hi"}]})
    assert t["tokens"] == ids("[user]hi[assistant]")
    status, _, d = request(port, "POST", "/detokenize", {"tokens": ids("ok") + [EOS]})
    assert d == {"prompt": "ok<eos>"}
    status, _, out = request(port, "POST", "/tokenize", {"prompt": "x", "bogus": 1})
    assert status == 400


def test_keep_alive_serves_several_requests_on_one_connection(server):
    port, _ = server()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    for _ in range(3):
        c.request("POST", "/v1/chat/completions", body=json.dumps(chat(max_tokens=2)).encode(),
                  headers={"Content-Type": "application/json"})
        r = c.getresponse()
        raw = r.read()
        assert r.status == 200 and raw.endswith(b"data: [DONE]\n\n")
    c.close()


# --------------------------------------------------------------------------------------------------
# pure helpers
# --------------------------------------------------------------------------------------------------
def test_incremental_detokenizer_holds_back_partial_utf8():
    text = "héllo € 😀!"
    d = serve.IncrementalDetokenizer(ByteTokenizer())
    deltas = [d.add(b) for b in text.encode("utf-8")]
    deltas.append(d.flush())
    assert "".join(deltas) == text
    assert not any("�" in x for x in deltas)


def test_pipelined_keeps_lookahead_in_flight_and_stops_on_close():
    log, in_flight, peak = [], set(), [0]

    def launch(i):
        log.append(("launch", i))
        in_flight.add(i)
        peak[0] = max(peak[0], len(in_flight))

    def collect(i):
        assert i in in_flight
        in_flight.discard(i)
        log.append(("collect", i))
        return i * 10

    assert list(serve.pipelined(10, 3, launch, collect)) == [i * 10 for i in range(10)]
    assert peak[0] == 3
    # step i + 3 is launched right after step i is collected and before it is yielded
    assert log[:4] == [("launch", 0), ("launch", 1), ("launch", 2), ("collect", 0)]
    assert log[4] == ("launch", 3)
    log.clear()
    in_flight.clear()
    g = serve.pipelined(100, 4, launch, collect)
    for _ in range(5):
        next(g)
    g.close()
    assert max(i for kind, i in log if kind == "launch") == 8   # 5 collected + 4 in flight
    assert sum(1 for kind, _ in log if kind == "launch") == 9


def test_config_line_round_trip():
    cfg = {"settings": {"pdl": True, "prefill_graphs": False}, "observed": {"kv_cache_dtype": "bfloat16"}}
    line = serve.format_config_line(cfg)
    assert "\n" not in line and line.startswith("[serve] resolved_config ")
    assert serve.parse_config_line("2026-10-13 " + line) == cfg
    assert serve.parse_config_line("[serve] listening") is None
    # the shared custom-engine format: the runner finds '{"inferencekit_resolved_config"' and parses the rest
    shared = serve.format_shared_config_line(cfg)
    assert "\n" not in shared
    i = shared.find('{"inferencekit_resolved_config"')
    assert i == 0 and json.loads(shared[i:])["inferencekit_resolved_config"] == cfg


def test_worker_survives_a_failing_drain(server):
    class BadClose(FakeModel):
        def stream(self, prompt_ids, max_tokens, logprobs, info):
            try:
                yield from super().stream(prompt_ids, max_tokens, logprobs, info)
            finally:
                if len(self.calls) == 1:
                    raise RuntimeError("drain failed")

    port, model = server(BadClose(TOKENS))
    _, _, events = sse(port, "/v1/chat/completions", chat(max_tokens=3))
    assert events[-1][1] == "[DONE]" and len(events) == 4   # the response was complete before the drain
    status, _, body = request(port, "POST", "/v1/chat/completions", chat(stream=False, max_tokens=2))
    assert status == 200 and body["choices"][0]["message"]["content"] == "He"
