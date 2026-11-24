#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
revision=2b1aee9fdd7d285b35292a9a73cc4f02c94bb65c
checkout="$root/upstream-adreno-llms"
if [[ ! -d $checkout/.git ]]; then
    git init -q "$checkout"
    git -C "$checkout" remote add origin https://github.com/a8nova/adreno-llms.git
    git -C "$checkout" fetch --depth=1 origin "$revision"
    git -C "$checkout" checkout --quiet --detach FETCH_HEAD
fi
test "$(git -C "$checkout" rev-parse HEAD)" = "$revision"
patch="$root/source/patches/adreno-kokoro-stream-tokenizer.patch"
if ! git -C "$checkout" apply --reverse --check "$patch" >/dev/null 2>&1; then
    git -C "$checkout" apply "$patch"
fi

ndk=${ANDROID_NDK_HOME:-"$root/toolchain/android-ndk-r27d"}
llvm="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
export ANDROID_NDK="$ndk"
export ADRENO_LLMS_CACHE="$root/build/adreno-deps"
export INFERENCEKIT_ESPEAK_BUILD=${INFERENCEKIT_ESPEAK_BUILD:-"$root/build/espeak"}
bash "$checkout/scripts/ci/setup_opencl_stub.sh"
bash source/build_espeak_android.sh
espeak="$INFERENCEKIT_ESPEAK_BUILD/espeak-ng-ed530aa113046142eb5115cf2fc9157854d0ffe1"
static="$root/build/espeak-adreno21"
# Match the upstream Adreno runner's Android API 21.
cmake -G Ninja -S "$espeak" -B "$static" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-21 -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_ESPEAK_NG_EXE=OFF -DBUILD_ESPEAK_NG_TESTS=OFF \
    -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_ASYNC=OFF -DUSE_SPEECHPLAYER=OFF
cmake --build "$static" --target espeak-ng --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"

model="$checkout/src/models/kokoro-82m"
dependency="$model/src/third_party/espeak-ng"
mkdir -p "$dependency/include"
cp -a "$espeak/src/include/espeak-ng" "$dependency/include"
printf 'CREATE %s\nADDLIB %s\nADDLIB %s\nSAVE\nEND\n' "$dependency/libespeak-ng.a" \
    "$static/src/libespeak-ng/libespeak-ng.a" "$static/src/ucd-tools/libucd.a" | "$llvm/llvm-ar" -M
NNOPT_DTYPE=fp16 bash "$model/scripts/build.sh"

kit="$root/build/kokoro-adreno-kit"
mkdir -p "$kit/weights" "$kit/assets"
payload="$model/build/fp16"
"$llvm/llvm-strip" --strip-all -o "$kit/Kokoro_82M_inference_fp16" "$payload/Kokoro_82M_inference_fp16"
"$llvm/llvm-strip" --strip-all -o "$kit/libclblast.so" "$payload/_deps/clblast-build/libclblast.so"
cp "$model/assets/phoneme_vocab.tsv" "$kit/assets/"
cp -a "$model/kernels" "$kit/"

url=https://huggingface.co/a8nova/adreno-llms-weights/resolve/7a5f7a1d4fe10589d4b7984b6161591065902707/kokoro-82m
while read -r digest source destination; do
    if [[ ! -f "$kit/$destination" ]]; then
        curl -fLsS --retry 2 "$url/$source" -o "$kit/$destination.partial"
        mv "$kit/$destination.partial" "$kit/$destination"
    fi
    printf '%s  %s\n' "$digest" "$kit/$destination" | sha256sum --check
done <<'ASSETS'
94725a9a8f70300e5fc600523bd664960f0fd0126b8ad9349ac28bdb04b77f7c model.fp16.bin weights/model.fp16.bin
355b6522bf36c14fc40e618e97f075b3c8c397cd5e6bbca18a2d1529468d39f9 model.fp16.meta.json weights/model.fp16.meta.json
d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b voices/af_heart.bin assets/voice_pack_af_heart.bin
ASSETS
