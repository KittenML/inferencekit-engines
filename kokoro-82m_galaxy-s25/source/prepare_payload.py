"""Fetch immutable upstream model assets on the Device Farm test host."""
import concurrent.futures
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
VIENEU = (
    "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/resolve/"
    "61b85e3d937fbbacb387714180e8182823512523/gguf"
)
SEA_G2P = (
    "https://raw.githubusercontent.com/pnnbao97/sea-g2p/"
    "e825173f235d08ea19315b2b279fb11153b44cea/python/sea_g2p/sea_g2p.bin"
)
LITERT = (
    "https://huggingface.co/litert-community/Kokoro-82M/resolve/"
    "bf1c2405642aa310bd210205c22d3ae1abc253fb"
)
ARTIFACTS = (
    ("kokoro-fp32", f"{SHERPA}/tts-models/kokoro-multi-lang-v1_0.tar.bz2",
     "60703707bc05ccd48d60e1eb723f38b07d6e474795a0b6a588d7220400e0f7a4"),
    ("kokoro-int8", f"{SHERPA}/tts-models/kokoro-int8-multi-lang-v1_0.tar.bz2",
     "72c347230107ef8a31098ec4d468a605244beef5f1ed00db97efa13518800653"),
    ("sherpa-runtime", f"{SHERPA}/v1.13.6/sherpa-onnx-v1.13.6-android.tar.bz2",
     "bb7f891b259f4faee5c55d4cc10d79a3ba2b27416e9320d2c4d044400834d75b"),
    ("vieneu/vieneu-v3-turbo-bf16.gguf", f"{VIENEU}/vieneu-v3-turbo-bf16.gguf",
     "c9c23d51989382e27730077c2373023bcfb0891db63a1efec97fd73b4bd6b7dc"),
    ("vieneu/vieneu-v3-turbo-q8_0.gguf", f"{VIENEU}/vieneu-v3-turbo-q8_0.gguf",
     "a60569e5d7dd6f24cbb7bb2e7c472bb988ad19563a4e78f0cc67cc53fd189bb4"),
    ("vieneu/ref_codes.txt", f"{VIENEU}/voices/minh_quan_pro/ref_codes.txt",
     "3e4e64aaaf657e68e906e16fdb7955e596f5b4ff94ef77dbd2313469312e5b2d"),
    ("vieneu/speaker.emb.txt", f"{VIENEU}/voices/minh_quan_pro/speaker.emb.txt",
     "f7bdb7235f5e3c14f9a45e60da8700df9b05eb6f880adf1cc143458c3c12b281"),
    ("vieneu/sea_g2p.bin", SEA_G2P,
     "4346e690d0711ebc5231e7a42c5c88aaf6e40377e894b4617c018fd81c6f4096"),
    ("mediapipe/kokoro-CPU.litertlm", f"{LITERT}/kokoro_int8_fp16.litertlm",
     "e7b931ee5d4b7104012a838c9eee240d19ea69cb3ca269041a3274f83dbb7fa1"),
    ("mediapipe/kokoro-GPU.litertlm", f"{LITERT}/kokoro_int8_gpu_fp16.litertlm",
     "1ef178b127ec48c737daf17eabb4cea5f6b3ffbc510aced270fe5c267bbeac52"),
)


def digest_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for buffer in iter(lambda: stream.read(1 << 20), b""):
            digest.update(buffer)
    return digest.hexdigest()


def fetch(root, scratch, artifact):
    name, url, expected = artifact
    temporary = scratch / name.replace("/", "_")
    subprocess.run(("curl", "-fLsS", "--connect-timeout", "15", "--max-time", "180",
                    "--retry", "2", "--output", str(temporary), url), check=True)
    if digest_of(temporary) != expected:
        raise RuntimeError(f"{name}: upstream asset SHA256 mismatch")
    print(f"{name}: SHA256 verified", flush=True)
    if name == "sherpa-runtime":
        subprocess.run(("tar", "-xf", str(temporary), "-C", str(scratch)), check=True)
        directory = next(scratch.rglob("arm64-v8a"))
        for filename in ("libsherpa-onnx-c-api.so", "libonnxruntime.so"):
            shutil.copyfile(directory / filename, root / "lib" / filename)
    elif name.startswith("kokoro-"):
        subprocess.run(("tar", "-xf", str(temporary), "-C", str(root / "models")), check=True)
    else:
        shutil.move(temporary, root / "models" / name)


def prepare(root, arms):
    wanted = {arm[1] for arm in arms}
    keys = set(wanted)
    if wanted & {"kokoro-audiocpp-fp32", "kokoro-adreno-fp16"}:
        keys.add("kokoro-fp32")
    if "vieneu-bf16" in wanted:
        keys.add("vieneu/vieneu-v3-turbo-bf16.gguf")
    if "vieneu-q8" in wanted:
        keys.add("vieneu/vieneu-v3-turbo-q8_0.gguf")
    if wanted & {"vieneu-bf16", "vieneu-q8"}:
        keys.update(("vieneu/sea_g2p.bin", "vieneu/ref_codes.txt", "vieneu/speaker.emb.txt"))
    for delegate in ("CPU", "GPU"):
        if f"kokoro-mediapipe-{delegate.lower()}" in wanted or (
                delegate == "GPU" and wanted & {"kokoro-mediapipe-gpu-bundle-on-cpu",
                                                "kokoro-mediapipe-gpu-123", "kokoro-mediapipe-npu-123",
                                                "kokoro-litert-omni-gpu"}):
            keys.add(f"mediapipe/kokoro-{delegate}.litertlm")
    # The runner links both runtimes even when only one family has measured arms.
    keys.add("sherpa-runtime")
    (root / "models/vieneu").mkdir(parents=True, exist_ok=True)
    (root / "models/mediapipe").mkdir(parents=True, exist_ok=True)
    (root / "lib").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as store, concurrent.futures.ThreadPoolExecutor(max_workers=6) as workers:
        futures = [workers.submit(fetch, root, Path(store), asset) for asset in ARTIFACTS if asset[0] in keys]
        for future in futures:
            future.result()
