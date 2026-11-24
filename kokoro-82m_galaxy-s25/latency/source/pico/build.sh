#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
runtime_revision=79769e32c3018cded68a48c58cde41cab07fbbb5
model_revision=a5632deac7d92f6178db814d933eba746a97f2dc
precision=${PICO_PRECISION:-int8}
case "$precision" in int8|fp32) ;; *) echo 'PICO_PRECISION must be int8 or fp32' >&2; exit 2;; esac
: "${ANDROID_NDK:?Set ANDROID_NDK to the Android NDK r27d directory}"

mkdir -p "$root/build" "$root/downloads" "$root/assets/models/pico"
if ! test -d "$root/upstream/.git"; then
    git init -q "$root/upstream"
fi
if ! git -C "$root/upstream" cat-file -e "$runtime_revision^{commit}" 2>/dev/null; then
    timeout 240 git -C "$root/upstream" fetch --depth=1 https://github.com/KittenML/kitten-inference "$runtime_revision"
fi
git -C "$root/upstream" checkout -q --detach "$runtime_revision"

(
    cd "$root/downloads"
    while read -r sha file; do
        if ! sha256sum --check --status <<< "$sha  $file"; then
            mkdir -p "$(dirname "$file")"
            curl --fail --silent --show-error --location --retry 2 --max-time 240 \
                "https://huggingface.co/KittenML/kitten-tts-2-pico/resolve/$model_revision/$file" -o "$file.tmp"
            sha256sum --check <<< "$sha  $file.tmp"
            mv "$file.tmp" "$file"
        fi
    done < "$root/model-sha256.txt"
)

ANDROID_API=24 KT08_SHARED=OFF "$root/upstream/engines/lite/mobile/build_android.sh" "$root/build/native"
compiler="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android24-clang++"
"$compiler" -std=c++20 -O3 -Wall -Wextra -Werror -fPIC -fvisibility=hidden \
    -static-libstdc++ -fopenmp -static-openmp -Wl,--exclude-libs,ALL -Wl,--gc-sections \
    -Wl,--no-undefined -Wl,-z,max-page-size=16384 -s -shared "-DPICO_PRECISION=\"$precision\"" \
    -I"$root/build/native/include" "$root/pico_jni.cpp" "$root/build/native/lib/libkitten_tts08.a" \
    -ldl -llog -o "$root/build/libpico_bench.so"

uv venv --allow-existing --python 3.12.13 "$root/build/venv"
python="$root/build/venv/bin/python"
timeout 600 uv pip sync --python "$python" "$root/python-lock.txt"
"$python" "$root/upstream/src/kitten_inference/tts2_tiny/convert.py" "$root/downloads" \
    "$root/build/converted" --model kitten_tts_2_pico_en
assets="$root/assets/models/pico"
if [[ "$precision" == int8 ]]; then
    rm -f "$assets/kitten_tts_2_pico_en_fp32.bin" "$assets/kitten_tts_2_pico_en_fp32_arch.json"
else
    rm -f "$assets/kitten_tts_2_pico_en_int8.bin" "$assets/kitten_tts_2_pico_en_int8_arch.json"
fi
cp "$root/build/converted/kitten_tts_2_pico_en_${precision}.bin" \
   "$root/build/converted/kitten_tts_2_pico_en_${precision}_arch.json" "$root/downloads/vocab.json" "$assets/"
cp "$root/build/converted/voices/en/eleanor.bin" "$assets/"
cp -R "$root/downloads/g2p" "$assets/"
"$python" "$root/upstream/src/kitten_inference/tts2_tiny/frontend/export_phone_data.py" "$assets/g2p"

(
    cd "$root"
    find assets/models/pico -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum build/libpico_bench.so
) > "$root/build/package-sha256.txt"
