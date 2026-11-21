#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
cd "$root"
profile=${1:-vieneu-cpp}
case "$profile" in
    vieneu-cpp|baseline) ;;
    *) echo 'Select vieneu-cpp or baseline.' >&2; exit 2 ;;
esac
if [[ $profile == kokoro-opencl ]]; then
    bash source/build_mediapipe_apk.sh
fi
if [[ $profile == kokoro-opencl ]]; then
    exit 0
fi
for command in git curl cmake ninja python3 cargo rustup unzip; do
    command -v "$command" >/dev/null || { printf 'Required command missing: %s\n' "$command" >&2; exit 1; }
done
if [[ $(uname -sm) != 'Linux x86_64' ]]; then
    echo 'Build on Linux x86-64 with the Android NDK.' >&2
    exit 1
fi
mkdir -p toolchain dependencies
ndk="${ANDROID_NDK_HOME:-$root/toolchain/android-ndk-r27d}"
if [[ -z ${ANDROID_NDK_HOME:-} && ! -d $ndk ]]; then
    zip="$root/toolchain/android-ndk-r27d-linux.zip"
    curl -fLsS --retry 2 https://dl.google.com/android/repository/android-ndk-r27d-linux.zip -o "$zip"
    printf '22105e410cf29afcf163760cc95522b9fb981121  %s\n' "$zip" | sha1sum -c -
    unzip -q "$zip" -d toolchain
    rm "$zip"
fi
if ! grep -qx 'Pkg.Revision = 27.3.13750724' "$ndk/source.properties"; then
    echo 'Set ANDROID_NDK_HOME to the pinned Android NDK r27d or leave it unset.' >&2
    exit 1
fi
export ANDROID_NDK_HOME="$ndk"

if [[ $profile == vieneu-cpp ]]; then
    rustup target add aarch64-linux-android
    python3 source/prepare_local.py vieneu-voice
    mkdir -p build/vieneu-cpp
    (
        cd build/vieneu-cpp
        bash "$root/source/competitors/vieneu-tts-cpp/reproduce.sh" "$ndk" \
            "$root/source/capture.cpp" "$root/build/local-payload/models/vieneu"
    )
    exit 0
fi

checkout() {
    local remote=$1 path=$2 revision=$3
    if [[ ! -d $path/.git ]]; then
        git init -q "$path"
        git -C "$path" remote add origin "$remote"
    fi
    if [[ $(git -C "$path" rev-parse HEAD 2>/dev/null || true) != "$revision" ]]; then
        git -C "$path" fetch -q --depth=1 origin "$revision"
        git -C "$path" checkout --quiet --detach FETCH_HEAD
    fi
}
checkout https://github.com/k2-fsa/sherpa-onnx.git upstream-sherpa 1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911
checkout https://github.com/0xShug0/audio.cpp.git upstream-audiocpp 749fffec6ad4a763520388a4240cd24e08a0a2bf
checkout https://github.com/pnnbao97/sea-g2p.git upstream-g2p e825173f235d08ea19315b2b279fb11153b44cea

python3 source/prepare_local.py runtime
bash source/build_audiocpp.sh stock
rustup target add aarch64-linux-android
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
cargo build --locked --manifest-path upstream-g2p/Cargo.toml --release \
    --target aarch64-linux-android --no-default-features --features capi --jobs "${INFERENCEKIT_BUILD_JOBS:-4}"
bash source/build_capture.sh
