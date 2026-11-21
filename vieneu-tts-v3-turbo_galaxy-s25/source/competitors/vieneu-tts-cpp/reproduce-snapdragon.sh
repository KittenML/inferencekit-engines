#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ${2:-scalar} != scalar && ${2:-scalar} != mixed ]]; then
    echo "Usage from a completed reproduce.sh build directory: $0 ANDROID_NDK [scalar|mixed]" >&2
    exit 2
fi

recipe_root=$(cd "$(dirname "$0")" && pwd)
ndk=$(realpath "$1")
root="$PWD"
llvm="$ndk/toolchains/llvm/prebuilt/linux-x86_64"
release="$root/snapdragon/llama-b12503"
test -x android-kit/capture-vieneu-cpp
test -f capture.cpp

release_digest=23fa926c45eec849b7fa6e51aa7e5094d70b3ebac7234330eb2e006118a950c1
if ! printf '%s  %s\n' "$release_digest" snapdragon.tar.gz | sha256sum --check --status; then
    curl -fsSL --retry 2 https://github.com/ggml-org/llama.cpp/releases/download/b12503/llama-b12503-bin-android-arm64-snapdragon.tar.gz \
        -o snapdragon.tar.gz.partial
    printf '%s  %s\n' "$release_digest" snapdragon.tar.gz.partial | sha256sum --check
    mv snapdragon.tar.gz.partial snapdragon.tar.gz
fi
mkdir -p snapdragon
tar -xzf snapdragon.tar.gz -C snapdragon

if ! git -C upstream apply --reverse --check "$recipe_root/snapdragon.patch" >/dev/null 2>&1; then
    git -C upstream apply "$recipe_root/snapdragon.patch"
fi
source_tree=upstream
capture=capture.cpp
layout=android-snapdragon
build=build-snap
if [[ ${2:-scalar} == mixed ]]; then
    source_tree=source-snapdragon-mixed
    capture="$source_tree/capture.cpp"
    layout=android-snapdragon-mixed
    build=build-snap-mixed
    mkdir -p "$source_tree"
    cp -R upstream/{src,tests,tools} "$source_tree/"
    cp upstream/CMakeLists.txt capture.cpp "$source_tree/"
    patch -d "$source_tree" -p1 --fuzz=0 < "$recipe_root/mixed-placement.patch"
fi
cmake -S "$source_tree" -B "$build" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DVIENEU_PORTABLE_CPU=ON \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
    -DONNXRUNTIME_ROOT="$root/ort" -DVIENEU_SEA_G2P=ON \
    -DVIENEU_LLAMA_DIR="$root/upstream/third_party/llama.cpp" \
    -DVIENEU_SEA_G2P_DIR="$root/upstream/third_party/sea-g2p" \
    -DVIENEU_ANDROID_G2P_LIB="$root/rust/aarch64-linux-android/release/libsea_g2p_rs.so" \
    -DVIENEU_ANDROID_LLAMA_KIT="$release"
cmake --build "$build" --target vieneu-tts --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"

mkdir -p "$layout/lib"
cp -alf android-kit/model "$layout/"
ln -f android-kit/sea_g2p.bin "$layout/"
cp "$build/libvieneu-tts.so" "$layout/lib/"
for name in libllama.so libggml.so libggml-base.so libggml-cpu.so libggml-rpc.so \
            libggml-opencl.so libggml-hexagon.so libggml-htp-v79.so; do
    cp "$release/lib/$name" "$layout/lib/"
done
for name in libc++_shared.so libonnxruntime.so libomp.so libsea_g2p_rs.so; do
    ln -f "android-kit/lib/$name" "$layout/lib/"
done
"$llvm/bin/aarch64-linux-android28-clang++" -std=c++17 -O3 -Wall -Wextra -Werror -pthread \
    -I "$source_tree/src" "$capture" -L "$layout/lib" -lvieneu-tts -Wl,-rpath,'$ORIGIN/lib' \
    -o "$layout/capture-vieneu-cpp"
"$llvm/bin/llvm-strip" --strip-unneeded "$layout/lib/libvieneu-tts.so"
find "$layout" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum > "$layout.sha256"
