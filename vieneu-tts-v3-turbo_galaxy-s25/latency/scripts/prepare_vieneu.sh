#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
study=$PWD
: "${ANDROID_NDK_HOME:?path to Android NDK r27d}"
ndk=$(realpath "$ANDROID_NDK_HOME")
llvm="$ndk/toolchains/llvm/prebuilt/linux-x86_64"
code=838979faf0354ffb3dff898e30b709f644fa7db4
model=61b85e3d937fbbacb387714180e8182823512523
native_model=a7b4f40050d4ff6d5225d8fd4fe6d571913844a2
codec=ceff0d0749bfb3fa2d61149794ec6feef0d1e1ae
work="$study/build/third-party"
payload="$study/build/payload/vieneu"
vendor="$study/vendor/VieNeu-TTS.cpp"
mkdir -p "$work" "$payload/codec" "$payload/acoustic" "$study/vendor"
download() {
    local url=$1 output=$2 digest=$3
    if [[ -f $output ]] && printf '%s  %s\n' "$digest" "$output" | sha256sum --check --status; then return; fi
    mkdir -p "$(dirname "$output")"
    curl -fsSL --retry 2 --max-time 900 "$url" -o "$output.partial"
    printf '%s  %s\n' "$digest" "$output.partial" | sha256sum --check
    mv "$output.partial" "$output"
}
apply_once() {
    if ! git -C "$1" apply --reverse --check "$2" 2>/dev/null; then git -C "$1" apply "$2"; fi
}
if [[ ! -d $vendor/.git ]]; then git init -q "$vendor"; fi
if [[ $(git -C "$vendor" rev-parse HEAD 2>/dev/null || true) != "$code" ]]; then
    git -C "$vendor" fetch --depth 1 https://github.com/dduongtrandai/VieNeu-TTS.cpp.git "$code"
    git -C "$vendor" checkout --detach FETCH_HEAD
fi
git -C "$vendor" submodule update --init --depth 1 third_party/llama.cpp third_party/sea-g2p
sea="$vendor/third_party/sea-g2p"
apply_once "$vendor" "$study/patches/android-g2p.patch"
apply_once "$sea" "$study/patches/cargo-lock.patch"
apply_once "$vendor" "$study/patches/native-streaming.patch"
apply_once "$vendor" "$study/official_onnx/compatibility.patch"
cp native/moss_step.h "$vendor/src/vieneu/v3_native/bench_moss_step.h"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$llvm/bin/aarch64-linux-android28-clang"
export CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS='-C link-arg=-Wl,-soname,libsea_g2p_rs.so'
export CARGO_TARGET_DIR="$study/build/rust"
cargo build --locked --manifest-path "$sea/Cargo.toml" --target aarch64-linux-android --release
cp "$sea/python/sea_g2p/sea_g2p.bin" "$payload/"
while read -r digest file; do
    case "$file" in
        ort.aar)
            download "https://repo.maven.apache.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.27.0/onnxruntime-android-1.27.0.aar" "$work/$file" "$digest"
            ;;
        quantizer.tar.gz)
            download "https://github.com/ggml-org/llama.cpp/releases/download/b12503/llama-b12503-bin-ubuntu-x64.tar.gz" "$work/$file" "$digest"
            ;;
        native-model/backbone.gguf)
            download "https://huggingface.co/lastudio-community/VieNeu-TTS-v3-Turbo-CPP/resolve/$native_model/backbone.gguf" "$work/$file" "$digest"
            ;;
        native-model/acoustic/*|native-model/codec/moss_audio_tokenizer_decode*|native-model/config.json|native-model/tokenizer.json|native-model/vieneu_v3_heads.npz)
            path="${file#native-model/}"
            download "https://huggingface.co/lastudio-community/VieNeu-TTS-v3-Turbo-CPP/resolve/$native_model/$path" "$payload/$path" "$digest"
            ;;
    esac
done < patches/downloads.sha256
while read -r digest file; do
    case "$file" in
        codec/*) download "https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX/resolve/$codec/${file#codec/}" "$payload/$file" "$digest" ;;
        onnx_int8/*) download "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/resolve/$model/$file" "$study/build/payload/vieneu-int8/${file#onnx_int8/}" "$digest" ;;
        voices/*) download "https://huggingface.co/pnnbao-ump/VieNeu-TTS-v3-Turbo/resolve/$model/gguf/$file" "$study/assets/$file" "$digest" ;;
    esac
done < assets/publisher.sha256
quant_sha=8f86dffde281672e974ed21c2ab51f706e0eca012af0fc934bc36d4bc8950fd1
if ! printf '%s  %s\n' "$quant_sha" "$payload/backbone.gguf" | sha256sum --check --status 2>/dev/null; then
    mkdir -p "$work/quantizer"
    tar -xzf "$work/quantizer.tar.gz" -C "$work/quantizer"
    quantizer="$work/quantizer/llama-b12503"
    LD_LIBRARY_PATH="$quantizer" "$quantizer/llama-quantize" "$work/native-model/backbone.gguf" "$payload/backbone.gguf" Q8_0
    printf '%s  %s\n' "$quant_sha" "$payload/backbone.gguf" | sha256sum --check
fi
python3 scripts/prepare_voice.py "$study/assets/voices/adam" "$payload/voices_v3_turbo.json"
"${VOICE_PREP_PYTHON:-python3}" official_onnx/prepare_voice.py "$study/build/payload/vieneu-int8" \
    "$study/assets/voices/adam" "$study/build/payload/vieneu-int8/voices.json"
mkdir -p "$work/ort/include" "$work/ort/lib"
unzip -oqj "$work/ort.aar" 'headers/*' -d "$work/ort/include"
unzip -oqj "$work/ort.aar" 'jni/arm64-v8a/libonnxruntime.so' -d "$work/ort/lib"
cmake -S "$vendor" -B build/native -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DVIENEU_PORTABLE_CPU=ON \
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
    -DONNXRUNTIME_ROOT="$work/ort" -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm \
    -DVIENEU_SEA_G2P=ON -DVIENEU_ANDROID_G2P_LIB="$CARGO_TARGET_DIR/aarch64-linux-android/release/libsea_g2p_rs.so"
cmake --build build/native --target vieneu-tts --parallel "${BUILD_JOBS:-8}"
cp workload.tsv build/payload/
