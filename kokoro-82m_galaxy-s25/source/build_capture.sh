#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
ndk=${ANDROID_NDK_HOME:-"$root/toolchain/android-ndk-r27d"}
bin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
reference="$root/dependencies/jniLibs/arm64-v8a"
mkdir -p build/stripped
"$bin/aarch64-linux-android28-clang++" -std=c++17 -DNDEBUG -O3 -Wall -Wextra -Werror \
    -static-libstdc++ -I upstream-sherpa -I upstream-audiocpp/include source/capture.cpp \
    -L "$reference" -L build-audio-android/bin -l sherpa-onnx-c-api -l audiocpp -Wl,-rpath,'$ORIGIN' \
    -o build/capture
"$bin/llvm-strip" --strip-all -o build/stripped/capture build/capture
"$bin/llvm-strip" --strip-all -o build/stripped/libaudiocpp.so build-audio-android/bin/libaudiocpp.so
"$bin/llvm-strip" --strip-all -o build/stripped/libsea_g2p_rs.so \
    upstream-g2p/target/aarch64-linux-android/release/libsea_g2p_rs.so
