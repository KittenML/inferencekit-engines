#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 ANDROID_NDK STUDY_CAPTURE_CPP" >&2
    exit 2
fi

ndk=$(realpath "$1")
capture=$(realpath "$2")
recipe=$(cd "$(dirname "$0")" && pwd)
llvm="$ndk/toolchains/llvm/prebuilt/linux-x86_64"
root="$PWD"
revision=59b9d7902ea62dbbbfb95e139e2bf0e8957f8753
if [[ ! -d upstream/.git ]]; then
    git init -q upstream
    git -C upstream fetch -q --depth=1 https://github.com/soniqo/speech-core.git "$revision"
    git -C upstream checkout -q --detach FETCH_HEAD
fi
test "$(git -C upstream rev-parse HEAD)" = "$revision"
mkdir -p ort/include ort/lib/arm64-v8a android-kit/lib
if [[ ! -f ort.aar ]]; then
    curl -fLsS --retry 2 \
        https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.27.0/onnxruntime-android-1.27.0.aar \
        -o ort.aar
fi
printf '077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb  ort.aar\n' | sha256sum --check
unzip -oqj ort.aar 'headers/*' -d ort/include
unzip -oqj ort.aar 'jni/arm64-v8a/libonnxruntime.so' -d ort/lib/arm64-v8a
cmake -G Ninja -S upstream -B build \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DSPEECH_CORE_BUILD_TESTS=OFF \
    -DSPEECH_CORE_WITH_ONNX=ON -DORT_DIR="$root/ort"
cmake --build build --target speech_core_models --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"
python3 "$recipe/make_capture.py" "$capture" capture.cpp
"$llvm/bin/aarch64-linux-android28-clang++" -std=c++17 -O3 -DNDEBUG -static-libstdc++ \
    -I upstream/include -I ort/include capture.cpp -L build -L ort/lib/arm64-v8a \
    -lspeech_core_models -lspeech_core -lonnxruntime -llog -ldl -Wl,-rpath,'$ORIGIN/lib' \
    -o capture-soniqo-kokoro
"$llvm/bin/llvm-strip" --strip-all -o android-kit/capture-soniqo-kokoro capture-soniqo-kokoro
cp ort/lib/arm64-v8a/libonnxruntime.so android-kit/lib/

checkpoint=https://huggingface.co/soniqo/Kokoro-82M-ONNX/resolve/a49a07fb772d5281a42570e8645bf238f7f73bf2
while read -r digest filename; do
    target="android-kit/model/$filename"
    if [[ -f $target ]] && printf '%s  %s\n' "$digest" "$target" | sha256sum --check --status; then
        continue
    fi
    mkdir -p "$(dirname "$target")"
    curl -fLsS --retry 2 "$checkpoint/$filename" -o "$target.partial"
    printf '%s  %s\n' "$digest" "$target.partial" | sha256sum --check
    mv "$target.partial" "$target"
done < "$recipe/model.sha256"
