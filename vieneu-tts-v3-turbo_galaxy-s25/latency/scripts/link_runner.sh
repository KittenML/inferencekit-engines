#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${ANDROID_NDK_HOME:?path to Android NDK r27d}"
ll="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64"
mkdir -p build/runner
cp build/native/bin/*.so build/native/libsea_g2p_rs.so build/third-party/ort/lib/libonnxruntime.so \
   "$ll/lib/clang/18/lib/linux/aarch64/libomp.so" \
   "$ll/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" build/runner/
"$ll/bin/aarch64-linux-android28-clang++" -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Werror \
    -fPIE -pie -fopenmp -pthread \
    -I pico -I pico/build/android/include -I official_onnx -I vendor/VieNeu-TTS.cpp/src \
    -I vendor/VieNeu-TTS.cpp/third_party/llama.cpp/include -I vendor/VieNeu-TTS.cpp/third_party/llama.cpp/ggml/include \
    -I vendor/VieNeu-TTS.cpp/third_party/llama.cpp/vendor -I build/third-party/ort/include \
    native/runner.cpp build/native/libvieneu-tts-core.a pico/build/android/lib/libkitten_tts08.a \
    -L build/runner -lllama -lggml -lggml-base -lggml-cpu -lonnxruntime -lsea_g2p_rs -laaudio -ldl -llog \
    -Wl,-rpath,'$ORIGIN' -Wl,--no-undefined -o build/runner/libstudy_runner.so
"$ll/bin/llvm-strip" --strip-unneeded build/runner/*.so
"$ll/bin/llvm-readelf" -d build/runner/libstudy_runner.so | rg 'NEEDED|RUNPATH'
