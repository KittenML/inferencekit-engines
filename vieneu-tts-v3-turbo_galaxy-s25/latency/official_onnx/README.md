# Publisher INT8 graphs on Android

This adapter runs the publisher's INT8 graphs with the Adam preset and the MOSS streaming decoder. Apply `compatibility.patch` to VieNeu-TTS.cpp `838979faf0354ffb3dff898e30b709f644fa7db4` alongside the lane's native patches. The patch uses the `bench_moss_step.h` staged by `scripts/prepare_vieneu.sh`. Preset enrollment is outside this adapter.

```bash
uv run --no-project --python 3.12 --with numpy==2.2.6 python official_onnx/prepare_voice.py \
  assets/vieneu-int8 assets/voices/adam build/payload/vieneu-int8/voices.json
```

Include `official_onnx_engine.h` and link the rebuilt `libvieneu-tts-core.a`. Pass the packaged INT8 graph directory, shared MOSS codec directory, and projected preset JSON to `OfficialOnnxEngine`. It defaults to four frames for the first PCM chunk and six thereafter. This port uses the lane's English frontend and text partitioning. The host parity check feeds the resulting phonemes to the publisher's Python engine.

```bash
VIENEU_HOST_BUILD=/path/to/vieneu-cpp-host-build ORT_HOST_LIB=/path/to/libonnxruntime.so.1.27.0 \
  ./official_onnx/host_build.sh
cp build/payload/vieneu-int8/voices.json official_onnx/build/voices.json
HF_HUB_OFFLINE=1 PYTHONPATH=vendor/VieNeu-TTS/src uv run --no-project --python 3.12 \
  --with numpy==2.2.6 --with onnxruntime==1.27.0 --with tokenizers==0.22.2 \
  python official_onnx/verify.py --host-build /path/to/vieneu-cpp-host-build --all
```

[Compatibility patch](compatibility.patch) · [Host parity check](verify.py) · [Publisher INT8 graphs](https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/tree/61b85e3d937fbbacb387714180e8182823512523/onnx_int8)
