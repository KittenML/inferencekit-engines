# VieNeu-TTS-v3-Turbo on Galaxy S25

On-device Vietnamese speech via the native VieNeu-TTS.cpp Q8_0 runtime, with a separate low-latency decoder experiment.

## Requirements

Galaxy S25 with USB debugging. Build on Linux x86-64 with Git, CMake, Ninja, Rust, Python 3.11+, curl, unzip, Android platform-tools and NDK r27d. Allow 8 GB for builds and weights. The optional latency benchmark also needs JDK 17 and Android SDK 35 with Build Tools 35.0.0.

## Run

```bash
./install.sh
./serve.sh
./test.sh
```

The smoke test saves a Vietnamese WAV under `build/output/`. Run `./bench.sh vieneu-sustained-cool` to repeat the capacity workload.

## Results

The native CPU runtime sustained **five Vietnamese streams for 7.5 minutes** with a 700 ms simulated startup buffer. At three streams with a 250 ms buffer it delivered 3.32 audio seconds per second on time versus 2.33 for the publisher’s audio.cpp in a single matched test. [Results and baseline details](REPORT.md)

## How it was made

Codex used InferenceKit to tune CPU threads per session and benchmark streaming with fixed release times. The separate single-user experiment decodes an initial three-frame group through the MOSS codec and adapts later groups to the queued audio. [Implementation and evidence](REPORT.md#details)

## Limits

The default profile uses the measured concurrency runtime. The separate adaptive decoder failed an exact-audio check and remains opt-in through its reproduction harness. Its confirmation latency numbers use English prompts. Cross-runtime perceptual equivalence is unproven.

## License

Apache-2.0. Downloaded models and upstream runtimes use their own licenses. [Pinned sources](latency/SOURCES.md)
