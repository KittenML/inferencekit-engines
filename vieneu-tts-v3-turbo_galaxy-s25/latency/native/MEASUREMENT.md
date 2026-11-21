# Android request measurements

Request timing starts immediately before the engine receives text. The runner uses a pre-opened AAudio float mono stream at 48 kHz for both engines. Pico's 24 kHz speech is upsampled by causal linear interpolation for playback. The PCM clock starts with the first source packet. Synthesis completion and device presentation of the last PCM frame are separate endpoints. Audio Real Time Factor (RTF) divides synthesis wall time by produced audio duration.

First audible-level audio is the start of the first 5 ms output window with RMS of at least −45 dBFS. The callback records the stream frame index. The monitor projects its presentation time from the first valid AAudio monotonic timestamp at or up to 250 ms after that frame. Missing anchors are null. This measures device presentation for digital samples and does not measure acoustic loudness or speaker-to-listener propagation. The Java runner captures media mute, volume and audio devices. [Android AAudio reference](https://developer.android.com/ndk/reference/group/audio)

The playback callback inserts silence when an engine packet arrives late. The runner reports gaps between PCM packets, excluding tail padding, and reports AAudio xruns separately. Voice and quality WAVs retain the engines' native sample rates.

The native runner samples `smaps_rollup` PSS and RSS every 100 ms and at request boundaries. These are sampled peaks for the engine process, including playback buffers and instrumentation. The kernel's `VmHWM` covers the full process lifetime. [Linux procfs documentation](https://docs.kernel.org/filesystems/proc.html)

The helper compiled and linked with Android NDK r27d for arm64 API 28. The host shim tests onset indexing, 24 kHz conversion, gaps, silent/empty speech, unavailable timestamps and memory sampling. The fixture timestamps in [the mock log](../logs/measurement-host-mock.jsonl) come from the shim.

```bash
# Run from the study directory, with ANDROID_NDK_HOME pointing to r27d.
mock_dir=$(mktemp -d)
mkdir "$mock_dir/aaudio"
cp "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/aaudio/AAudio.h" \
    "$mock_dir/aaudio/"
g++ -std=c++17 -O2 -Wall -Wextra -Werror -pthread -fsanitize=address,undefined \
    '-D__INTRODUCED_IN(x)=' -D_Nonnull= -D_Nullable= -I "$mock_dir" \
    native/measurement_mock.cc -o "$mock_dir/mock"
"$mock_dir/mock"
```
