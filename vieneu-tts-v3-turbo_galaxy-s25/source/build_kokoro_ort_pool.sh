#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
ndk=${ANDROID_NDK_HOME:-"$root/toolchain/android-ndk-r27d"}
work="$root/build/kokoro-ort-pool"
mkdir -p "$work/headers" build/stripped/ort-pool
archive="$work/onnxruntime-android-1.27.1.zip"
if [[ ! -f $archive ]]; then
    curl -fLsS --retry 2 \
        https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.1/onnxruntime-android-1.27.1.zip \
        -o "$archive"
fi
printf 'defade26209f72cf4fa9769b18052c842833d6bef12924595d26f03b995548ca  %s\n' "$archive" | sha256sum --check
unzip -oqj "$archive" 'headers/*.h' -d "$work/headers"
export SHERPA_ONNXRUNTIME_INCLUDE_DIR="$work/headers"
export SHERPA_ONNXRUNTIME_LIB_DIR="$root/dependencies/jniLibs/arm64-v8a"
printf 'dc5e4c172b1be9e530c6a62ad8f1be3e0a911cabdee6195abf28dab72477e194  %s\n' \
    "$SHERPA_ONNXRUNTIME_LIB_DIR/libonnxruntime.so" | sha256sum --check

revision=1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911
overlay="$work/sherpa"
if [[ ! -e $overlay/.git ]]; then
    git -C upstream-sherpa worktree add --detach "$overlay" "$revision"
fi
patch="$root/source/patches/kokoro-ort-global-pool.patch"
if ! git -C "$overlay" apply --reverse --check "$patch" >/dev/null 2>&1; then
    git -C "$overlay" apply "$patch"
fi
cmake -S "$overlay" -B "$work/build" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_TTS=ON -DSHERPA_ONNX_ENABLE_C_API=ON \
    -DSHERPA_ONNX_ENABLE_BINARY=OFF -DSHERPA_ONNX_ENABLE_JNI=OFF \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DBUILD_PIPER_PHONMIZE_EXE=OFF -DBUILD_PIPER_PHONMIZE_TESTS=OFF \
    -DBUILD_ESPEAK_NG_EXE=OFF -DBUILD_ESPEAK_NG_TESTS=OFF
cmake --build "$work/build" --target sherpa-onnx-c-api --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"
"$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" --strip-unneeded \
    -o "$root/build/stripped/ort-pool/libsherpa-onnx-c-api.so" "$work/build/lib/libsherpa-onnx-c-api.so"
