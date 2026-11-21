# Kitten TTS 2 Pico Android adapter

`PicoEngine` synthesizes English text with `eleanor_en` through the Kitten streaming API and delivers mono float PCM at 24 kHz. It runs native INT8 or FP32. The adapter defaults to 64-character chunks and Android's low-memory mode.

```bash
ANDROID_NDK=/path/to/android-ndk-r27d ./pico/prepare.sh
```

Package `pico/build/assets` as one directory. Include `pico/build/android/include` and link `pico/build/android/lib/libkitten_tts08.a` with `-fopenmp -ldl`. When linking alongside an engine that already uses shared OpenMP, use the same NDK `libomp.so` for both. The recipe uses Android NDK r27d, API level 28, CMake, Ninja, and `uv`.

[Kitten inference source](https://github.com/KittenML/kitten-inference/tree/79769e32c3018cded68a48c58cde41cab07fbbb5/engines/lite/mobile) · [Pico model and frontend assets](https://huggingface.co/KittenML/kitten-tts-2-pico/tree/da8c9d029e16d9302f1827adb303d92316e25f73) · [Android NDK downloads](https://developer.android.com/ndk/downloads)
