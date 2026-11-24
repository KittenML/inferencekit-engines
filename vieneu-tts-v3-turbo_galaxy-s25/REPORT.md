# VieNeu-TTS-v3-Turbo on Galaxy S25

## Result

One CPU thread per VieNeu-TTS.cpp Q8_0 session sustained **four simultaneous Vietnamese streams with a 250 ms startup buffer** and **five with 700 ms**. Each passing cell lasted 7.5 minutes. The buffers are simulated playback settings. They do not change model inference time. Five streams with 250 ms still had eight gaps. [Sustained timing and playback replay](evidence/capacity/vieneu-sustained-cool/)

| Three simultaneous Vietnamese streams, 250 ms simulated buffer | First audio p95 | On-time requests | Audio generated on time | Simulated gaps |
| --- | ---: | ---: | ---: | ---: |
| VieNeu-TTS.cpp Q8_0, one thread/session | 1.34 s | 450/450 | 3.32 s/s | 0 |
| Publisher’s audio.cpp Q8_0, two threads/session | 5.66 s | 319/450 | 2.33 s/s | 227 |

This single matched capacity cell delivered **1.43× as much audio on time**. The native packages use different model conversions and began at Android thermal status 0 and 1 respectively. Same-runtime parity passed on retained VieNeu-TTS.cpp output. Cross-runtime speech equivalence is unproven. [Raw timings, rate calculation and PCM evidence](evidence/capacity/vieneu-sustained-matched/)

At 250 ms, changing from two CPU threads per session to one increased the verified capacity **from three to four simultaneous streams** in four balanced repeats. Both thread settings completed every c4 request on time. The one-thread setting eliminated two gaps in the two-thread control, and all 168 comparable saved chunks matched byte for byte. At one stream the measured two- versus four-thread onset interval included parity. [Matched thread tests](evidence/capacity/vieneu-native-confirmation/)

### Sentence delivery at one and three streams

The one-thread native VieNeu-TTS.cpp Q8_0 profile also ran four matched Vietnamese rounds at one and three streams with sentence and whole-utterance delivery. Both modes passed the playback simulation. The whole-utterance to sentence startup latency ratio was **2.18× [95% CI 2.15–2.19]** at one stream and **2.05× [2.00–2.09]** at three. These are ratios of median worst-stream p95 first nonzero PCM for the native profile. [Matched timing](evidence/capacity/whole-matched-controls/vieneu-cpp-whole-vs-sentences-paired.json) · [Run plan](evidence/capacity/whole-matched-controls/run-plan.json)

### One active request with Android playback

The experimental decoder retains the native Q8_0 talker, acoustic model and voice. It decodes the first three codec frames early, then uses smaller or larger groups depending on the buffered audio. The paired first-speech ratios below compare it against the full MOSS decoder on the same text, runtime and playback policy. Warm confirmation used four CPU threads and six matched **English** rounds. [Latency summary](evidence/single-user/confirmation-summary.json)

| Prompt lengths | Full-decoder first audio p95 | Adaptive first audio p95 | Paired onset ratio [95% CI] |
| --- | ---: | ---: | ---: |
| Short | 1,974 ms | 335 ms | 4.70× [3.85, 5.60] |
| Medium | 1,304 ms | 238 ms | 4.52× [4.01, 5.28] |
| Long | 1,142 ms | 349 ms | 2.71× [2.52, 3.19] |

Full synthesis took longer at the median in each bucket with the candidate. Twelve retained pairs had identical lengths with at most a one-count PCM16 difference, which **fails the registered bit-exact criterion**. Digital playback logged no gaps in the confirmation. The retained Vietnamese clips check waveform and speech quality, with no matched Vietnamese onset benchmark in this experiment. The adaptive candidate’s multi-stream capacity remains untested. [Quality and diagnostic outputs](evidence/single-user/confirmation-quality.json) · [Continuity checks](evidence/single-user/confirmation-checks.json)

## Run it

After [installing the engine](README.md#run), use the pinned InferenceKit importer and test the registered capacity phases.

```bash
uv tool install 'git+https://github.com/KittenML/inferencekit.git@b23245653361e3619ba91825b080b1e7942c5682'
./bench.sh vieneu-sustained-matched
./bench.sh vieneu-sustained-cool
./bench.sh vieneu-sustained-frontier
./bench.sh vieneu-native-confirmation
./bench.sh whole-matched-controls
```

Build the adaptive decoder APK using the [build instructions](latency/REPRODUCE.md) and run `./bench.sh playback latency/build/s25-study.apk results/one-request`. This selects the `confirmation` plan. Use labels `full-t4-eager` and `adaptive3-t4-eager` for the paired reducer. The default `install.sh`, `serve.sh` and `test.sh` use the native capacity profile.

## Files and outputs

- [Pinned native Android runtime build](source/competitors/vieneu-tts-cpp/reproduce.sh), [audio.cpp baseline build](source/build_audiocpp.sh), [one-request source pins](latency/SOURCES.md)
- [Adaptive codec code](latency/native/moss_step.h), [native publisher patch](latency/patches/native-streaming.patch), [Android presentation clock](latency/native/MEASUREMENT.md)
- [Capacity evidence](evidence/capacity/), [single-user capture and receipts](evidence/single-user/capture/), [replay protocol](latency/PROTOCOL.md)

## Details

One CPU thread per session reduces competition between concurrent workers. In the four-stream comparison the rate of produced audio and request completions stayed the same, while the one-thread setting removed the two underruns that made the two-thread configuration fail. The five-stream capacity result adds a longer startup buffer. A six-stream sustained run missed deadlines even when given a playback buffer. [Thread sweep](evidence/capacity/vieneu-native-confirmation/) · [Sustained limit](evidence/capacity/vieneu-sustained-frontier/)

The adaptive path carries the streaming MOSS decoder’s state between calls and emits audio before all codec frames have been produced. Starting with three frames reduces initial work. Small groups refill a low queue, and larger groups reduce per-call overhead once there is audio queued. The saved timing and fidelity results apply to this experimental profile. [Patch and codec implementation](latency/patches/native-streaming.patch)
