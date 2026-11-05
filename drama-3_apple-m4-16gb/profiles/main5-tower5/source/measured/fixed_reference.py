# SPDX-License-Identifier: Apache-2.0
"""Strict single-asset canonical-code cache for the separate conditioned trial.

No encoder is enabled. The frozen manifest, WAV, transcript and integer code
array must all match; every other reference is refused.
"""
import hashlib
import io
import importlib
import json
from pathlib import Path
from types import MethodType
import wave

CONTRACT_SHA256 = "3195ac5bd26cdea22324ca47cfbca82f6acc00f0fddf1ae4d153c8754e964359"
REFERENCE_TEXT = "The quiet room was warm, and a small lamp lit the desk."


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _child(root, name):
    if not isinstance(name, str):
        raise ValueError("registered artifact path must be a string")
    value = (root / name).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("registered artifact must remain beside its manifest")
    return value


def wav_pcm(raw):
    with wave.open(io.BytesIO(raw), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype()) != (44100, 1, 2, "NONE"):
            raise ValueError("fixed reference must be mono 44.1kHz PCM16")
        count = reader.getnframes()
        pcm = reader.readframes(count)
        if not 44100 <= count <= 30 * 44100 or len(pcm) != count * 2:
            raise ValueError("fixed reference must be complete and 1..30 seconds")
        return pcm, count


class FixedReference:
    def __init__(self, manifest, expected_sha256):
        import numpy as np
        self.np = np
        path = Path(manifest).resolve()
        raw = path.read_bytes()
        if digest(raw) != expected_sha256:
            raise ValueError("fixed-reference manifest digest mismatch")
        spec = json.loads(raw)
        if spec.get("schema") != "inferencekit-fixed-reference-v1" or spec.get("contract_sha256") != CONTRACT_SHA256:
            raise ValueError("unregistered conditioning contract")
        if spec.get("asset_admission_verdict") != "PASS" or not spec.get("asset_admission_evidence_sha256"):
            raise ValueError("asset has no sealed machine admission proof")
        if not spec.get("producer_resolved") or not spec.get("canonical_codec_provenance"):
            raise ValueError("asset/codec producer provenance missing")
        self.text = spec["reference_text"]
        if self.text != REFERENCE_TEXT or digest(self.text.encode()) != spec["reference_text_utf8_sha256"]:
            raise ValueError("reference transcript digest mismatch")
        wav = _child(path.parent, spec["wav_file"]).read_bytes()
        pcm, count = wav_pcm(wav)
        if digest(wav) != spec["wav_sha256"] or digest(pcm) != spec["pcm_sha256"] or count != spec["sample_count"]:
            raise ValueError("registered reference waveform changed")
        codes_path = _child(path.parent, spec["codes_file"])
        if digest(codes_path.read_bytes()) != spec["codes_file_sha256"]:
            raise ValueError("canonical code file digest mismatch")
        codes = np.load(codes_path, allow_pickle=False)
        if codes.dtype != np.dtype("<i8") or codes.ndim != 2 or codes.shape[0] != 10 or not 1 <= codes.shape[1] <= 1024:
            raise ValueError("canonical codes must be int64 [10, frames]")
        codes = np.ascontiguousarray(codes, dtype="<i8")
        if list(codes.shape) != spec["codes_shape"] or int(codes.min()) < 0 or int(codes.max()) >= 4096:
            raise ValueError("canonical code shape/range mismatch")
        if digest(codes.tobytes(order="C")) != spec["codes_logical_int64le_sha256"]:
            raise ValueError("canonical logical codes changed")
        self.codes = codes.astype("<i4")
        self.mlx_codes_logical_int32le_sha256 = digest(self.codes.tobytes(order="C"))
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        self.samples_sha256 = digest(samples.tobytes())
        self.metadata = {key: spec[key] for key in (
            "wav_sha256", "pcm_sha256", "sample_count", "reference_text_utf8_sha256",
            "codes_file_sha256", "codes_shape", "codes_logical_int64le_sha256",
            "contract_sha256", "asset_admission_evidence_sha256")}
        self.metadata["manifest_sha256"] = expected_sha256
        self.metadata["mlx_codes_logical_int32le_sha256"] = self.mlx_codes_logical_int32le_sha256

    def validate_request(self, wav, text):
        pcm, count = wav_pcm(wav)
        if text != self.text or digest(wav) != self.metadata["wav_sha256"] or digest(pcm) != self.metadata["pcm_sha256"] or count != self.metadata["sample_count"]:
            raise ValueError("reference does not match the sole registered asset")

    def install(self, model, mx):
        registered = self
        codes = mx.array(self.codes, dtype=mx.int32)
        mx.eval(codes)

        def encode(this, samples):
            data = registered.np.asarray(samples, dtype=registered.np.float32).reshape(-1)
            if digest(data.tobytes()) != registered.samples_sha256:
                raise ValueError("fixed-reference code cache miss; encoder remains disabled")
            return codes

        object.__setattr__(model, "_encode_audio", MethodType(encode, model))

    def initial_prompt_proof(self, model, fixture_path, mx):
        """Build actual first-segment frames without model forward or codec encode."""
        raw = Path(fixture_path).read_bytes()
        split_text = importlib.import_module(model.__class__.__module__).split_text
        codes = mx.array(self.codes, dtype=mx.int32)
        rows = []
        seen = set()
        for line in raw.decode().splitlines():
            fixture = json.loads(line)
            if fixture["id"] in seen:
                raise ValueError("duplicate prompt fixture id")
            seen.add(fixture["id"])
            chunks = split_text(fixture["text"])
            if not chunks:
                raise ValueError("empty conditioned prompt")
            from segment_contract import partition_proof
            partition = partition_proof(fixture["text"], chunks)
            frames, system_rows = model.build_frames([(self.text, codes)], [], chunks[0])
            mx.eval(frames)
            logical = self.np.ascontiguousarray(self.np.asarray(frames), dtype="<i8")
            rows.append({"id":fixture["id"], "text_utf8_sha256":digest(fixture["text"].encode()),
                         "first_segment_text_utf8_sha256":digest(chunks[0].encode()),
                         "native_text_segment_count":len(chunks), "shape":list(logical.shape),
                         "system_rows":int(system_rows),
                         "native_segment_UTF8_sha256":[item['text_utf8_sha256'] for item in partition['segments']],
                         "ordered_text_partition":partition,
                         "logical_int64le_sha256":digest(logical.tobytes(order="C"))})
        return {"schema":"inferencekit-conditioned-initial-prompts-v1", "fixed_reference":self.metadata,
                "fixtures_file_sha256":digest(raw), "logical_layout":"row-major [T,11], signed int64 little endian",
                "model_forward_calls":0, "codec_encode_calls":0, "rows":rows}
