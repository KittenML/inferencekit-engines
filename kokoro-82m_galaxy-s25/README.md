# Kokoro-82M on Galaxy S25

On-device English sentence streaming with the MediaPipe LiteRT vocoder on Adreno OpenCL. An independent Adreno FP16 configuration and a Sherpa-ONNX CPU benchmark are included.

## Requirements

Galaxy S25 with USB debugging. Build on Linux x86-64 with JDK 17, Android SDK 35, Build Tools 35.0.0, platform-tools, Python 3.11+, curl and unzip. Set `ANDROID_SDK_ROOT`. Allow 8 GB of host disk space. The optional comparison builds also use NDK r27d, CMake, Ninja and Rust.

## Run

```bash
./install.sh
./serve.sh
./test.sh
```

The smoke test saves a WAV under `build/output/`. Run `./bench.sh kokoro-sustained` to repeat the capacity workload on an attached device.

## Results

MediaPipe INT8 sustained **three simultaneous streams for 7.5 minutes** with simulated playback. In the separate single-request AudioTrack test it finished synthesis in **2.18 s p95** with zero underruns. The standalone Adreno build also sustained three streams. Both failed the tested four-stream load. [Timings, baselines and audio checks](REPORT.md)

## How it was made

- **Adreno vocoder.** Codex (GPT-6.1 Terra) used InferenceKit to expose Android's vendor `libOpenCL.so` in the APK manifest and select MediaPipe's GPU delegate. LiteRT runs the vocoder, which turns acoustic features into PCM audio, on the Adreno GPU. [Android manifest](source/mediapipe/AndroidManifest.xml)
- **Warm sentence workers.** The sustained MediaPipe profile keeps one initialized model session on a dedicated worker per stream. Each worker reuses its session across requests and emits each sentence's PCM as it finishes. [MediaPipe worker](source/mediapipe/MediaPipeCapture.java)
- **Persistent native OpenCL runner.** The standalone path keeps an Adreno-LLMs worker alive across sentences. Streaming uses eSpeak phoneme IDs and skips loading the text tokenizer. The Android playback profile uses FP16 weights and an FP32 generator. [Streaming patch](source/patches/adreno-kokoro-stream-tokenizer.patch) · [JNI bridge](latency/source/kokoro/adreno_jni.cpp)
- **Overlapped playback.** The Android latency harness queues each completed PCM chunk to a high-priority audio thread. Non-blocking AudioTrack writes let playback proceed while synthesis works on the next sentence. [Playback code](latency/source/android/Playback.java)

## Limits

Capacity uses simulated playback. The FP32 CPU startup experiment measures Android presentation and incurs underruns on the longer prompts. Cross-runtime perceptual equivalence is unproven.

## License

Apache-2.0. Downloaded models and upstream runtimes use their own licenses. [Pinned sources](REPORT.md#files-and-outputs)
