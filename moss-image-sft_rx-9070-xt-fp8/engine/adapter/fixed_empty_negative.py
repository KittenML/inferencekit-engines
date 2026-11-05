"""Narrow routing around the pinned native prompt-cache API.

No model imports at module load. The optional installer is dependency injection
for pure source controls, not an alternate production cache implementation.
This helper does not modify or normalize any model input or tensor value.
"""
from contextlib import contextmanager
from types import SimpleNamespace


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class FixedEmptyNegativePolicy:
    def __init__(self, native_encode, *, model_tag, max_sequence_length=4096, installer=None):
        require(callable(native_encode), "Native bound encoder required")
        require(isinstance(model_tag, str) and model_tag, "Frozen model/config identity required")
        require(max_sequence_length == 4096, "Only the approved sequence-length policy is admitted")
        if installer is None:
            # Remote-only import; native owns storage, keying and detach behavior.
            from vllm_omni.diffusion.cache.prompt_embed_cache import install_prompt_embed_cache
            installer = install_prompt_embed_cache
        self.native_encode = native_encode
        self.max_sequence_length = max_sequence_length
        self.holder = SimpleNamespace(encode_prompt=native_encode)
        self.cache = installer(self.holder, max_size=1, enabled=True, model_tag=model_tag)
        require(self.cache is not None, "Native cache installer did not bind the encoder")
        self.cached_encode = self.holder.encode_prompt
        self._active = None
        self.completed_requests = 0
        self.positive_bypasses = 0
        self.negative_calls = 0
        self.last_call = None

    @contextmanager
    def request(self, positive_prompt):
        """Enter only for a validated real request, never a native dummy run."""
        require(self._active is None, "Concurrent/nested request cache routing is forbidden")
        require(isinstance(positive_prompt, str), "One exact positive string required")
        self._active = {"positive": positive_prompt, "calls": 0}
        self.last_call = None
        try:
            yield
            require(self._active["calls"] == 2, "Native request must encode positive then empty negative exactly once")
            self.completed_requests += 1
        finally:
            self._active = None

    def encode(self, prompt, max_sequence_length=4096):
        if self._active is None:
            # Native initialization/dummy profiling remains uncached. This must
            # never be used to authorize an unregistered served request.
            return self.native_encode(prompt, max_sequence_length)
        index = self._active["calls"]
        require(index in (0, 1), "Unexpected additional encoding in native request")
        require(max_sequence_length == self.max_sequence_length, "Encoding length policy changed")
        expected = self._active["positive"] if index == 0 else ""
        require(isinstance(prompt, list) and len(prompt) == 1 and prompt[0] == expected,
                "Actual text/order differs from fixed native request topology")
        before = self.cache.stats()
        if index == 0:
            result = self.native_encode(prompt, max_sequence_length)
            self.positive_bypasses += 1
            require(self.cache.stats() == before, "Positive encoding touched the native negative cache")
            branch = "positive"
        else:
            result = self.cached_encode(prompt, max_sequence_length)
            self.negative_calls += 1
            branch = "negative"
        after = self.cache.stats()
        require(after["max_size"] == 1 and after["size"] <= 1, "Native fixed cache capacity changed")
        if index == 1:
            require(after["size"] == 1 and after["bypassed"] == before["bypassed"], "Empty negative not cached natively")
            require((after["hits"] - before["hits"], after["misses"] - before["misses"]) in ((1, 0), (0, 1)),
                    "Unexpected native negative cache accounting")
        self._active["calls"] += 1
        self.last_call = {"branch": branch, "native_cache_before": before, "native_cache_after": after,
                          "actual_text_encoder_forwards": "Must be independently counted by the worker observer"}
        return result

    def stats(self):
        return {"native": self.cache.stats(), "positive_bypasses": self.positive_bypasses,
                "negative_calls": self.negative_calls, "completed_requests": self.completed_requests}
