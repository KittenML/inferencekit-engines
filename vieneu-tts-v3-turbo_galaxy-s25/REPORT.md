# VieNeu-TTS-v3-Turbo single-request playback on Galaxy S25

## Result

Six paired warm rounds on a base Galaxy S25 compared the full MOSS decoder with the adaptive decoder in VieNeu-TTS.cpp Q8_0. Both used four CPU threads, the same weights, voice, English text and playback policy. The test measured the first presented speech sample using Android’s digital playback clock. [Recorded timing](evidence/confirmation-summary.json)

| English prompt | Full decoder, first speech p95 | Adaptive decoder, first speech p95 | Paired onset ratio [95% CI] |
| --- | ---: | ---: | ---: |
| Short | 1,974 ms | 335 ms | 4.70× [3.85, 5.60] |
| Medium | 1,304 ms | 238 ms | 4.52× [4.01, 5.28] |
| Long | 1,142 ms | 349 ms | 2.71× [2.52, 3.19] |

The adaptive decoder took longer to finish synthesis at the median in all three buckets. Both modes logged zero digital playback gaps. Twelve retained pairs had equal lengths and a maximum difference of one PCM16 count, so the adaptive decoder **failed the registered bit-exact check**. Vietnamese clips were retained for quality checks. The matched latency figures use English prompts. [Quality checks](evidence/confirmation-quality.json) · [Playback diagnostics](evidence/confirmation-checks.json)

## Run it

Build and install the APK with the [README commands](README.md#run). Run the paired confirmation with `./bench.sh` and analyze it with the [replay commands](latency/REPRODUCE.md).

## Files and outputs

- [Pinned model and runtimes](latency/SOURCES.md)
- [Adaptive decoder](latency/native/moss_step.h) and [streaming patch](latency/patches/native-streaming.patch)
- [Timing and quality protocol](latency/PROTOCOL.md)
- [Sanitized single-request capture](evidence/single-request-capture.tar.gz)

## Details

The adaptive mode decodes the first three codec frames and carries the MOSS state into subsequent calls. It uses smaller groups when the playback queue runs low and larger groups as the queue fills. The full decoder is the default smoke test. The experimental mode is available through `./test.sh adaptive`, and the paired benchmark runs both modes.
