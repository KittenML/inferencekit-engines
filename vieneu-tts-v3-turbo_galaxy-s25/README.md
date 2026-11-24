# VieNeu-TTS-v3-Turbo on Galaxy S25

A single-request Android playback harness for an experimental adaptive MOSS decoder.

## Requirements

Galaxy S25 with USB debugging. On Linux x86-64, install JDK 17, Android SDK 35 and Build Tools 35.0.0, Android NDK r27d, Git, Rust, CMake, Ninja, curl, unzip, and Python 3.11+ with NumPy. Set `JAVA_HOME`, `ANDROID_SDK_ROOT`, and `ANDROID_NDK_HOME`. Put `adb` on PATH and allow 8 GB for the build and weights.

## Run

```bash
./install.sh
./serve.sh
./test.sh
```

The smoke test renders Vietnamese WAVs with the full decoder. Run `./test.sh adaptive` to try the experimental decoder. Run `./bench.sh` to repeat the paired single-request comparison.

## Results

On the Galaxy S25, the adaptive decoder started speech in **238–349 ms p95** across three English prompt lengths. The same VieNeu-TTS.cpp Q8_0 runtime with the full decoder took **1.14–1.97 s**. The adaptive decoder took longer to finish synthesis and failed the registered bit-exact PCM check. [Results and evidence](REPORT.md)

## How it was made

InferenceKit tested a streaming codec that decodes the first three frames early, carries decoder state forward, and adjusts later groups to refill the audio queue.

## Limits

The English timings use Android’s digital playback clock. Vietnamese clips were checked for output quality. The adaptive mode is experimental. [Quality results](evidence/confirmation-quality.json)

## License

Apache-2.0. Downloaded checkpoints and upstream runtimes use their own licenses. [Pinned sources](latency/SOURCES.md)
