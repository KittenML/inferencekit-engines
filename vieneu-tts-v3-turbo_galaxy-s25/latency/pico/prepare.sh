#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
upstream="$here/../vendor/kitten-inference"
revision=79769e32c3018cded68a48c58cde41cab07fbbb5
model_rev=da8c9d029e16d9302f1827adb303d92316e25f73
mode=${1:-all}
case "$mode" in
    assets|android|all) ;;
    *) printf 'Usage: %s [assets|android|all]\n' "$0" >&2; exit 2 ;;
esac

git init -q "$upstream"
if [[ $(git -C "$upstream" rev-parse HEAD 2>/dev/null || true) != "$revision" ]]; then
    git -C "$upstream" fetch -q --depth 1 https://github.com/KittenML/kitten-inference.git "$revision"
    git -C "$upstream" checkout -q --detach FETCH_HEAD
fi
if ! git -C "$upstream" diff --quiet HEAD; then
    printf 'Modified tracked files in %s\n' "$upstream" >&2
    exit 1
fi

if [[ $mode != android ]]; then
    model_url="https://huggingface.co/KittenML/kitten-tts-2-pico/resolve/$model_rev"
    downloads="$here/build/downloads"
    assets="$here/build/assets"
    mkdir -p "$downloads" "$assets"
    (
        cd "$downloads"
        while read -r digest file; do
            if printf '%s  %s\n' "$digest" "$file" | sha256sum --check --status; then
                continue
            fi
            mkdir -p "$(dirname "$file")"
            curl -fsSL --retry 3 --connect-timeout 20 --max-time 240 "$model_url/$file" -o "$file.partial"
            printf '%s  %s\n' "$digest" "$file.partial" | sha256sum --check
            mv "$file.partial" "$file"
        done < "$here/downloads.sha256"
        sha256sum --check "$here/downloads.sha256"
    )
    cp -a "$downloads/g2p" "$assets/"
    cp "$downloads/vocab.json" "$assets/"
    tiny="$upstream/src/kitten_inference/tts2_tiny"
    uv run --no-project --python 3.12 --with numpy==2.2.6 \
        python "$tiny/convert.py" "$downloads" "$here/build/converted" --model kitten_tts_2_pico_en
    cp "$here/build/converted/"kitten_tts_2_pico_en_{fp32,int8}{.bin,_arch.json} "$assets/"
    cp "$here/build/converted/voices/en/eleanor.bin" "$assets/"
    uv run --no-project --python 3.12 --with numpy==2.2.6 --with spacy==3.8.11 --with click==8.1.8 \
        python "$tiny/frontend/export_phone_data.py" "$assets/g2p"
    (
        cd "$assets"
        find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
    ) > "$here/build/assets.sha256"
fi

if [[ $mode != assets ]]; then
    : "${ANDROID_NDK:?set ANDROID_NDK to the Android NDK r27d directory}"
    rg -q '^Pkg.Revision = 27\.3\.13750724$' "$ANDROID_NDK/source.properties"
    ANDROID_API=28 KT08_SHARED=OFF bash "$upstream/engines/lite/mobile/build_android.sh" "$here/build/android"
fi
