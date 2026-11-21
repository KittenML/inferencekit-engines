#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 ANDROID_NDK STUDY_CAPTURE_CPP PUBLISHER_VOICE_DIR" >&2
    exit 2
fi

ndk=$(realpath "$1")
study_capture=$(realpath "$2")
voice_dir=$(realpath "$3")
llvm="$ndk/toolchains/llvm/prebuilt/linux-x86_64"
recipe_root=$(cd "$(dirname "$0")" && pwd)
export LC_ALL=C
git init -q upstream
revision=838979faf0354ffb3dff898e30b709f644fa7db4
if [[ $(git -C upstream rev-parse HEAD 2>/dev/null || true) != "$revision" ]]; then
    git -C upstream fetch -q --depth=1 https://github.com/dduongtrandai/VieNeu-TTS.cpp.git "$revision"
    git -C upstream checkout -q --detach FETCH_HEAD
fi
git -C upstream submodule update --init --depth 1 third_party/llama.cpp third_party/sea-g2p
apply_patch_once() {
    if ! git -C "$1" apply --reverse --check "$2" >/dev/null 2>&1; then
        git -C "$1" apply "$2"
    fi
}
apply_patch_once upstream "$recipe_root/android-g2p.patch"
sea_g2p=upstream/third_party/sea-g2p
# The published lockfile retained a build dependency removed from this pinned manifest.
apply_patch_once "$sea_g2p" "$recipe_root/cargo-lock.patch"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$llvm/bin/aarch64-linux-android28-clang"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS='-C link-arg=-Wl,-soname,libsea_g2p_rs.so'
export CARGO_TARGET_DIR="$PWD/rust"
cargo build --locked --manifest-path "$sea_g2p/Cargo.toml" --target aarch64-linux-android --release

while read -r digest filename; do
    if [[ -f $filename ]] && printf '%s  %s\n' "$digest" "$filename" | sha256sum --check --status; then
        continue
    fi
    case "$filename" in
        ort.aar)
            url=https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.27.0/onnxruntime-android-1.27.0.aar
            ;;
        quantizer.tar.gz)
            url=https://github.com/ggml-org/llama.cpp/releases/download/b12503/llama-b12503-bin-ubuntu-x64.tar.gz
            ;;
        native-model/*)
            url=https://huggingface.co/lastudio-community/VieNeu-TTS-v3-Turbo-CPP/resolve/a7b4f40050d4ff6d5225d8fd4fe6d571913844a2/${filename#native-model/}
            ;;
        *) exit 2 ;;
    esac
    mkdir -p "$(dirname "$filename")"
    curl -fsSL --retry 2 "$url" -o "$filename.partial"
    printf '%s  %s\n' "$digest" "$filename.partial" | sha256sum --check --status
    mv "$filename.partial" "$filename"
done < "$recipe_root/downloads.sha256"
sha256sum -c "$recipe_root/downloads.sha256"
mkdir -p ort/include ort/lib quantizer
unzip -oq -j ort.aar 'headers/*' -d ort/include
unzip -oq -j ort.aar 'jni/arm64-v8a/libonnxruntime.so' -d ort/lib
tar -xzf quantizer.tar.gz -C quantizer
root="$PWD"
backbone_sha=8f86dffde281672e974ed21c2ab51f706e0eca012af0fc934bc36d4bc8950fd1
if [[ ! -f backbone-q8_0.gguf ]] ||
    ! printf '%s  backbone-q8_0.gguf\n' "$backbone_sha" | sha256sum --check --status; then
    LD_LIBRARY_PATH="$root/quantizer/llama-b12503" \
        quantizer/llama-b12503/llama-quantize native-model/backbone.gguf backbone-q8_0.gguf Q8_0
    printf '%s  backbone-q8_0.gguf\n' "$backbone_sha" | sha256sum --check
fi

cmake -S upstream -B build -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DVIENEU_PORTABLE_CPU=ON \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
    -DONNXRUNTIME_ROOT="$root/ort" -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm \
    -DVIENEU_SEA_G2P=ON -DVIENEU_ANDROID_G2P_LIB="$CARGO_TARGET_DIR/aarch64-linux-android/release/libsea_g2p_rs.so"
cmake --build build --target vieneu-tts --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"
python3 "$recipe_root/make_capture.py" "$study_capture" capture.cpp

mkdir -p android-kit/lib android-kit/model
"$llvm/bin/aarch64-linux-android28-clang++" -std=c++17 -O3 -DNDEBUG -Wall -Wextra -Werror -pthread \
    -I upstream/src capture.cpp -L build -lvieneu-tts -Wl,-rpath,'$ORIGIN/lib' -o android-kit/capture-vieneu-cpp
cp build/libvieneu-tts.so build/libsea_g2p_rs.so build/bin/*.so ort/lib/libonnxruntime.so \
    "$llvm/lib/clang/18/lib/linux/aarch64/libomp.so" \
    "$llvm/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so" android-kit/lib/
"$llvm/bin/llvm-strip" --strip-unneeded android-kit/lib/*.so
cp "$sea_g2p/python/sea_g2p/sea_g2p.bin" android-kit/
cp -alf native-model/acoustic native-model/codec android-kit/model/
for filename in config.json tokenizer.json vieneu_v3_heads.npz speaker_encoder.onnx denoiser.onnx; do
    ln -f "native-model/$filename" "android-kit/model/$filename"
done
ln -f backbone-q8_0.gguf android-kit/model/backbone.gguf
python3 "$recipe_root/make_voice.py" native-model/voices_v3_turbo.json "$voice_dir/ref_codes.txt" \
    "$voice_dir/speaker.emb.txt" android-kit/model/voices_v3_turbo.json
