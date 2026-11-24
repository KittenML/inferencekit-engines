# Reproduce the single-request Galaxy S25 study

Set the Android and build environment described in the [engine README](../README.md#requirements). Use `VOICE_PREP_PYTHON` if NumPy is installed in a separate virtual environment. The APK contains the pinned full and adaptive VieNeu decoders and a Pico control used in the recorded study. [Sources and revisions](SOURCES.md)

From the engine directory, run the confirmation on a base Galaxy S25 with USB debugging.

```bash
./install.sh
./bench.sh results/confirmation
python3 latency/scripts/summarize.py results/confirmation --baseline full-t4-eager --candidate adaptive3-t4-eager
```

The APK launches a fresh process per cell and saves per-request JSONL and retained WAVs. For offline audio checks, install the scorer dependencies in [the protocol](PROTOCOL.md) and run:

```bash
python3 latency/scripts/score_quality.py results/confirmation --baseline full-t4-eager --candidate adaptive3-t4-eager
```

Use the pinned local Faster Whisper and UTMOS checkpoints. `--no-mos` runs the transcript and waveform checks alone. To replay the stored timings, extract [the sanitized capture](../evidence/single-request-capture.tar.gz) and pass its `capture` folder to the timing reducer.
