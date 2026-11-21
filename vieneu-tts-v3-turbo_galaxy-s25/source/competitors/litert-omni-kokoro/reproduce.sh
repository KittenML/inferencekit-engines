#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "Usage: $0 ANDROID_NDK BAZEL STUDY_CAPTURE_CPP" >&2
    exit 2
fi

export ANDROID_NDK_HOME="$(realpath "$1")"
bazel=$(realpath "$2")
capture=$(realpath "$3")
root="$PWD"
recipe=$(cd "$(dirname "$0")" && pwd)
llvm="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/bin"
export PATH="$llvm:$PATH" CC="$llvm/clang" CXX="$llvm/clang++"
revision=7a03724cece417f1724c923b3d527b4319730841
grep -qx 'Pkg.Revision = 28.1.13356709' "$ANDROID_NDK_HOME/source.properties"
test "$("$bazel" --version)" = 'bazel 7.6.1'
if [[ ! -d upstream/.git ]]; then
    git init -q upstream
    git -C upstream fetch -q --depth=1 https://github.com/google-ai-edge/LiteRT-LM.git "$revision"
    git -C upstream checkout -q --detach FETCH_HEAD
fi
test "$(git -C upstream rev-parse HEAD)" = "$revision"
cd upstream
for library in libLiteRtGpuAccelerator.so libLiteRtOpenClAccelerator.so libLiteRtWebGpuAccelerator.so libwebgpu_dawn.so; do
    path="prebuilt/android_arm64/$library"
    digest=$(git show "HEAD:$path" | sed -n 's/^oid sha256://p')
    if ! printf '%s  %s\n' "$digest" "$path" | sha256sum --check --status; then
        curl -fLsS --retry 2 "https://media.githubusercontent.com/media/google-ai-edge/LiteRT-LM/$revision/$path" \
            -o "$path.partial"
        printf '%s  %s\n' "$digest" "$path.partial" | sha256sum --check
        mv "$path.partial" "$path"
    fi
done
python3 "$recipe/make_capture.py" "$capture" omni/inferencekit_kokoro_capture.cc
cp "$(dirname "$capture")/kokoro_pcm_process.h" omni/kokoro_pcm_process.h
if grep -Fq 'name = "inferencekit_kokoro_capture"' omni/BUILD; then
    sed -i 's/srcs = \["inferencekit_kokoro_capture.cc"\],/srcs = ["inferencekit_kokoro_capture.cc", "kokoro_pcm_process.h"],/' omni/BUILD
else
    cat "$recipe/BUILD.capture" >> omni/BUILD
fi
"$bazel" --output_user_root="$root/bazel-cache" build --config=android_arm64 \
    --repo_env="CC=$CC" --jobs="${INFERENCEKIT_BUILD_JOBS:-6}" //omni:inferencekit_kokoro_capture
mkdir -p "$root/android-kit/lib"
"$llvm/llvm-strip" --strip-all -o "$root/android-kit/capture-omni-kokoro" bazel-bin/omni/inferencekit_kokoro_capture
cp prebuilt/android_arm64/libLiteRt* prebuilt/android_arm64/libwebgpu_dawn.so "$root/android-kit/lib/"
