# Kokoro-82M on Galaxy S25

## Result

The default MediaPipe INT8 profile with an Adreno OpenCL vocoder played individual requests through Android AudioTrack without underruns and sustained three streams in a simulated playback workload. The standalone Adreno profile also passed both workloads.

| Kokoro profile | One active request with AudioTrack, p95 estimated first speech / full synthesis | Three streams for 7.5 minutes, audio generated per second / worst-stream p95 first nonzero PCM |
| --- | --- | --- |
| MediaPipe LiteRT INT8 with Adreno OpenCL vocoder | 1.46 s / 2.18 s, 0 of 12 requests with underruns | 4.61 s/s / 1.45 s, 0 simulated gaps |
| Standalone Adreno, FP16 weights and FP32 generator | 1.44 s / 2.39 s, 0 of 12 requests with underruns | 5.56 s/s / 1.23 s, 0 simulated gaps |

The phone playback runs and capacity runs used different request schedules and onset measurements. Each sustained three-stream run finished all 450 requests on time. Both profiles failed the sustained four-stream load. [Android playback results](evidence/measurements.json) · [MediaPipe capacity](evidence/benchmark-archive.tar.gz) · [Standalone Adreno capacity](evidence/benchmark-archive.tar.gz)

Four matched rounds also tested sentence delivery and whole-utterance delivery within the same profile at one and three streams. For MediaPipe, the whole-utterance to sentence startup latency ratio was **1.65× [95% CI 1.64–1.70]** at one stream and **1.74× [1.64–1.80]** at three. The standalone Adreno ratios were **2.14× [2.09–2.32]** and **1.53× [1.48–1.73]**. These are ratios of median worst-stream p95 first nonzero PCM. Both modes passed the playback simulation. The standalone Adreno profile finished requests sooner with whole-utterance delivery. [MediaPipe paired latency](evidence/measurements.json) · [Adreno paired latency](evidence/measurements.json) · [Full-response timing](evidence/measurements.json)

### One request against published Kokoro checkpoints

The same base Galaxy S25 generated the same three sentence chunks for each English request. A shared Android playback timer measured all four configurations. The MediaPipe and standalone Adreno logs confirmed OpenCL placement in every matched round. Pico kept its native leading pause. [Timing rows and placement](evidence/measurements.json) · [Signed APK and checkpoint hashes](evidence/measurements.json)

| Deployment | First PCM p95 | First speech p95 | Full synthesis p95 | Requests with underruns | Retained ASR WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sherpa-ONNX published Kokoro INT8 v1.0, CPU | 7.81 s | 8.11 s | 16.26 s | 12/12 | 1.0% |
| MediaPipe 1.2.2 LiteRT INT8, Adreno OpenCL vocoder | 1.06 s | 1.46 s | 2.18 s | 0/12 | 0% |
| Independent Adreno build, FP16 weights and FP32 generator | 1.04 s | 1.44 s | 2.39 s | 0/12 | 2.9% |
| Kitten TTS 2 Pico native INT8, CPU | 0.32 s | 0.74 s | 0.72 s | 0/12 | 0% |

Three complete matched rounds contribute 12 requests per row. The host timeout cut off the final Adreno cell and prevented the last MediaPipe cell from running. Both completed CPU runs from that round are retained in the raw evidence. These measurements describe the tested deployments and voices. The GPU rows use the two optimized Kokoro profiles from the capacity study. [Capture and host exit](evidence/measurements.json) · [Quality scores](evidence/measurements.json)

The published Sherpa graph has 88 quantized convolutions, 83 integer MatMuls and six dynamically quantized LSTMs. It retains floating-point operations too. The earlier local MatMul-only control in the precision matrix is a distinct configuration. [Packaged model audit](evidence/measurements.json)

The four retained clips per deployment matched their PCM hashes and the recognizer recovered every prompt ending. UTMOSv2 scored the two CPU arms. It returned a scorer error for both GPU arms, including when scored separately. Human listening and cross-runtime naturalness equivalence remain unverified. [Sherpa naturalness](evidence/measurements.json) · [Pico naturalness](evidence/measurements.json) · [MediaPipe diagnostic](evidence/measurements.json) · [Adreno diagnostic](evidence/measurements.json)

### Which standard baselines ran?

| Android runtime tested | Measured outcome |
| --- | --- |
| Sherpa-ONNX 1.13.6 FP32 CPU, two threads | One stream held in the earlier short workload, with 811–909 ms p95 first audio across repeats. It failed at two streams. The later matched Kitten CPU run used longer first sentences and Android presentation timing. |
| MediaPipe Tasks Audio 1.2.2, LiteRT INT8 with Adreno OpenCL vocoder | 658 ms p95 first signal in the matched one-stream Omni screen, and three sustained streams in its separate capacity run. |
| Standalone LiteRT-LM Omni 0.19.1, same INT8 bundle and Adreno OpenCL | 872 ms p95 at one stream in that screen. At higher loads, most workers emitted silent PCM. The 658/872 ms comparison is a single short screen, and cross-runtime waveform equivalence is unproven. |
| Adreno-LLMs FP16 OpenCL | Three sustained streams in the independent capacity run. A four-stream result on a short workload did not carry over to the longer test. |
| Soniqo FP16 ONNX CPU and NNAPI requested, MediaPipe 1.2.3 and isolated Omni | Device deployment attempted. The capped job stopped before timings exported. Timing **NOT RUN** for comparison purposes. |

[CPU qualification](evidence/benchmark-archive.tar.gz) · [Omni head-to-head](evidence/benchmark-archive.tar.gz) · [Capped competing-runtime run](evidence/benchmark-archive.tar.gz)

### Kitten TTS 2 on the matched FP32 CPU workload

The strongest repeated CPU comparison used the same four English prompts and three sentence boundaries, the original Kitten Hugging Face FP32 weights, the seeded FP32 Kokoro graph in Sherpa-ONNX 1.13.6, and four CPU inference threads per deployment. Each ran its own voice and phonemizer. Six paired fresh-worker rounds used the same physical Galaxy S25 and AudioTrack path. Mini and Kokoro each have about 82 million parameters. [Exact text and sentence boundaries](latency/fixtures/matched-chunks.jsonl) · [FP32 weight audit](evidence/measurements.json) · [Six-round evidence](evidence/measurements.json)

| FP32 CPU deployment, four threads | Median first PCM | Median estimated first speech | Median full synthesis | Requests with AudioTrack underruns |
| --- | ---: | ---: | ---: | ---: |
| Kokoro-82M, Sherpa-ONNX | 2.790 s | 3.062 s | 7.219 s | 12/24 |
| Kitten TTS 2 Mini, Kitten Inference | 1.374 s | 1.799 s | 3.809 s | 0/24 |
| Kitten TTS 2 Pico, Kitten Inference | 179 ms | 575 ms | 494 ms | 0/24 |

The paired Kokoro-to-Mini ratio was **1.91× [95% CI 1.89–1.92]** for full synthesis and **1.74× [1.71–1.76]** for estimated first speech. The ratios divide the medians of the per-round medians. Speech recognition scored zero word errors on four Mini clips and 0.98% WER on four Kokoro clips. Listener-rated naturalness equivalence is **UNPROVEN**. The timing claim applies to these FP32 CPU deployments and prompts. [Request traces](evidence/measurements.json) · [Speech checks](evidence/measurements.json)

A separate one-round foreground screen pinned each four-thread deployment to the same four-core affinity mask, including the two prime cores. On the same four prompts, median synthesis took **3.62 s for Kokoro, 1.40 s for Mini and 178 ms for Pico**, with zero underruns in each arm. The observed Kokoro-to-Mini synthesis ratio was 2.58×. A six-thread Mini diagnostic reached 1.09 s and produced different waveform hashes from four-thread Mini. All 16 request timings were retained before the host timeout. The single foreground round gives a descriptive result. [Foreground summary](evidence/benchmark-archive.tar.gz) · [Request traces and affinity](evidence/benchmark-archive.tar.gz) · [Speech checks](evidence/benchmark-archive.tar.gz)

### Why playback ran out in the earlier CPU tests

The separate CPU matrix used one phone, two threads, identical English sentences and immediate Android playback. At FP32, p95 first speech was **0.84 s for Pico, 6.66 s for Mini and 7.08 s for Kokoro**. Pico had zero underruns, Mini had 27 of 32, and Kokoro had 32 of 32. Mini INT8 reached 2.75 s and Pico INT8 reached 0.76 s, both with zero underruns. Kokoro’s INT8 control quantizes eligible MatMul layers and retains FP32 convolutions. The measurements compare tested deployments and show each model’s own voice and runtime. [Full precision matrix](evidence/benchmark-archive.tar.gz)

Sentence streaming lets the player start before synthesis finishes. Playback stays continuous when enough audio is queued by the time each chunk starts. In the two-thread matrix, the median interval from the second chunk being ready to the third chunk being ready was **4.31 s for Kokoro and 4.30 s for Mini**. Their second chunks contained a median **1.75 s and 2.30 s** of audio respectively. Any audio left from the first chunk helped cover the wait. Once that queue emptied, AudioTrack logged an underrun. Smaller chunks can start earlier, and their per-chunk overhead and the time to generate subsequent audio still matter. In the four-thread repeat, Mini cut the median interval between chunks two and three to 1.37 s and had zero underruns. Kokoro still took 2.46 s and had underruns on half the requests. [Two-thread chunk timings](evidence/benchmark-archive.tar.gz) · [Four-thread playback traces](evidence/measurements.json)

The separate seeded two-thread Sherpa experiment isolated callback-triggered playback. It moved p95 speech presentation from 10.09 s to 5.17 s, with exact PCM in 36 pairs. Synthesis p95 was 9.74 s buffered and 9.72 s with callbacks. All 12 streamed two-sentence requests had an AudioTrack underrun. [Seeded timing](evidence/benchmark-archive.tar.gz) · [Exact PCM checks](evidence/benchmark-archive.tar.gz)

## Fresh Galaxy S25 recheck

The Android 16 test reproduced the three-stream result for 7.5 minutes. All 450 requests finished with zero simulated gaps and throughput was unchanged at 4.61 audio seconds per second. Worst-stream p95 first audio moved from 1.45 s to 1.64 s. The four-stream capture was interrupted before export.

On Android 15, two complete matched rounds gave 2.16 s p95 synthesis and zero AudioTrack underruns in eight MediaPipe requests. The archived comparison was 2.18 s in 12 requests. The full panel ran past the booking's capture timeout. The backends, timing rows and exact fixtures for the completed rounds are in the archive.

## Run it

After [installing the engine](README.md#run), use Android USB debugging and the pinned InferenceKit importer.

```bash
uv tool install 'git+https://github.com/KittenML/inferencekit.git@b23245653361e3619ba91825b080b1e7942c5682'
./bench.sh kokoro-sustained
./bench.sh kokoro-adreno-sustained
./bench.sh kokoro-adreno-confirmation
./bench.sh whole-matched-controls
```

The extra Adreno and CPU benchmark routes use Android NDK r27d, Git, CMake, Ninja, Rust and the NDK linker. The capacity workflows run this package’s model-specific arms of the original study. The archived run plans record the original cell order.

For the one-request AudioTrack trial, set `ANDROID_TOOLCHAIN` to the directory layout described in the [Android build script](latency/source/build.sh). Run `bench.sh playback` from this engine directory with a signed playback APK.

```bash
bash latency/reproduce.sh build seeded
./bench.sh playback latency/build/s25-tts-seeded.apk results/replay 28
```

The seeded APK replays Sherpa FP32 and the Pico control, both with warm engines. Extract the benchmark archive to replay the older precision matrix and published-checkpoint panel.

```bash
mkdir -p results/benchmark-archive
tar -xzf evidence/benchmark-archive.tar.gz -C results/benchmark-archive
cd results/benchmark-archive
python3 latency/analysis/precision_matrix.py evidence/physical-precision-matrix/results \
  latency/fixtures/matched-chunks.jsonl /tmp/s25-precision-summary.json
python3 latency/analysis/release_panel.py evidence/published-panel \
  latency/fixtures/matched-chunks.jsonl /tmp/s25-published-panel.json
cd ../..
```

The published-checkpoint playback panel uses `bash latency/reproduce.sh build release-panel` followed by `./bench.sh playback latency/build/s25-tts-release-panel.apk results/release-panel`. The full panel needs the pinned Android Adreno build in `ADRENO_KOKORO_KIT`, the verified checkpoint files under `latency/build/release-assets`, and the development Kitten native library and model assets. [APK build](latency/source/build.sh) · [Panel asset packer](latency/source/pack_release_panel.py) · [Asset manifest](evidence/measurements.json)

## Files and outputs

[Selected measurements](evidence/measurements.json) · [Sanitized archive and checksums](evidence/benchmark-archive.tar.gz)

- [Pinned MediaPipe dependencies](source/mediapipe/dependencies.sha256), [Adreno build](source/build_adreno_kokoro.sh), [Sherpa and model assets](source/prepare_payload.py)
- [Capacity scheduler](source/run_device.py), [one-request benchmark](evidence/benchmark-archive.tar.gz), [AudioTrack source](latency/source/android/Playback.java)
- [Raw capacity evidence](evidence/benchmark-archive.tar.gz), [single-request timings and hashes](evidence/benchmark-archive.tar.gz), [Kitten matched controls](evidence/benchmark-archive.tar.gz)

## Details

The MediaPipe APK exposes the vendor OpenCL library through its manifest. Device logs confirm that the vocoder initializes on Adreno. The independent FP16 integration exposes per-sentence callbacks and preserves phonemizer and voice handling. Disabling optional dot-product paths removed errors on one short screen. Four balanced repetitions established three passing streams for both settings and the latency interval included parity. [Android manifest](source/mediapipe/AndroidManifest.xml) · [Adreno patch](source/patches/adreno-kokoro-stream-tokenizer.patch) · [Repeat evidence](evidence/benchmark-archive.tar.gz)

The Sherpa JNI path forwards its existing complete-segment callback to a dedicated audio thread. Sentence boundaries, model and inference threads are fixed between the buffered and callback runs. Additional initial buffering can absorb temporary stalls, and sustained playback requires synthesis to supply audio at least as fast as playback consumes it. [JNI bridge](latency/source/kokoro/kokoro_jni.cpp) · [Queue and chunk traces](evidence/benchmark-archive.tar.gz)

The three-stream GPU runs use simulated playback. The published-checkpoint single-request panel uses Android presentation on the same sentence boundaries for all arms. The earlier CPU precision matrix and seeded Sherpa study are separate measurements. [Single-user fixture and timing protocol](evidence/benchmark-archive.tar.gz) · [Published-checkpoint panel](evidence/benchmark-archive.tar.gz)
