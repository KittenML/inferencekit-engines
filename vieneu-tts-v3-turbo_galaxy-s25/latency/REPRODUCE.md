# Reproduce the Galaxy S25 study

Use Linux x86-64 with JDK 17, Android SDK 35, Build Tools 35.0.0, Android NDK r27d, Rust with the `aarch64-linux-android` target, CMake, Ninja, Python 3 with NumPy, and the public model assets. [Sources and revisions](SOURCES.md)

```bash
export ANDROID_NDK_HOME=/path/to/android-ndk-r27d
export PHONE_BUILD_TOOLS=/path/to/android-sdk/build-tools/35.0.0
export PHONE_ANDROID_JAR=/path/to/android-sdk/platforms/android-35/android.jar
export JAVA_HOME=/path/to/jdk-17
scripts/prepare_vieneu.sh
pico/prepare.sh all
mkdir -p build/payload/pico
cp -a pico/build/assets/. build/payload/pico/
scripts/link_runner.sh
scripts/phone-build.sh build/s25-study.apk
```

Set `VOICE_PREP_PYTHON` if NumPy is installed in a virtual environment. Boot a physical Galaxy S25 running Android 15 and use the reproduced APK with the selected plan. The APK uses fresh runner processes and stores its per-request JSONL and retained WAVs in its internal files directory. [Phone launch and capture](app/REPRODUCE.md)

```bash
python3 scripts/summarize.py results/phone/confirmation --baseline full-t4-eager --candidate adaptive3-t4-eager
python scripts/score_quality.py results/phone/confirmation --baseline full-t4-eager --candidate adaptive3-t4-eager
```

Use the local Faster Whisper snapshots and SpeechMOS checkpoint pinned in the diagnostics. `--no-mos` runs only transcript and waveform checks. The scoring environment used faster-whisper 1.2.1, torch 2.6.0 and torchaudio 2.6.0. [Quality protocol](PROTOCOL.md)
