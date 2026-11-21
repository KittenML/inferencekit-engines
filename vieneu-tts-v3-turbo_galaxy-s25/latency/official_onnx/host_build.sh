#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
: "${VIENEU_HOST_BUILD:?point to a host CMake build of pinned VieNeu-TTS.cpp with SEA-G2P enabled}"
: "${ORT_HOST_LIB:?point to the Linux ONNX Runtime 1.27 shared library}"
support=$(realpath "$VIENEU_HOST_BUILD")
ort=$(realpath "$ORT_HOST_LIB")
llama=$(realpath ../vendor/VieNeu-TTS.cpp/third_party/llama.cpp)
revision=838979faf0354ffb3dff898e30b709f644fa7db4
git init -q upstream
if [[ $(git -C upstream rev-parse HEAD 2>/dev/null || true) != "$revision" ]]; then
    git -C upstream fetch --quiet --depth 1 https://github.com/dduongtrandai/VieNeu-TTS.cpp.git "$revision"
    git -C upstream checkout --quiet --detach FETCH_HEAD
fi
if ! git -C upstream apply --reverse --check "$here/compatibility.patch" 2>/dev/null; then
    git -C upstream apply "$here/compatibility.patch"
fi
mkdir -p build/lib
ln -sfn "$ort" build/lib/libonnxruntime.so
ln -sfn libonnxruntime.so build/lib/libonnxruntime.so.1

cp ../native/moss_step.h upstream/src/vieneu/v3_native/bench_moss_step.h
inc=(-I upstream/src -I ../build/third-party/ort/include -I "$llama/vendor" -I "$llama/include" -I "$llama/ggml/include")
sources=(smoke.cpp upstream/src/vieneu/v3_onnx/*.cpp)
objects=()
jobs=()
for source in "${sources[@]}"; do
    object="build/$(basename "$source").o"
    "${CXX:-g++}" -std=c++17 -O3 -pthread -fvisibility=hidden "${inc[@]}" -c "$source" -o "$object" &
    jobs+=("$!")
    objects+=("$object")
done
failed=0
for job in "${jobs[@]}"; do
    wait "$job" || failed=1
done
(( failed == 0 ))
"${CXX:-g++}" -pthread "${objects[@]}" -L "$support" -L "$support/bin" -L "$here/build/lib" \
    -Wl,-rpath,"$support:$support/bin:$here/build/lib" -lvieneu-tts -lggml-cpu -lggml-base -lonnxruntime -o build/smoke
