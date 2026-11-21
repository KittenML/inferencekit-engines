#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
revision=ed530aa113046142eb5115cf2fc9157854d0ffe1
ndk=${ANDROID_NDK_HOME:-"$root/toolchain/android-ndk-r27d"}
build=${INFERENCEKIT_ESPEAK_BUILD:-"$root/build/espeak"}
mkdir -p "$build" build/stripped/kokoro
archive="$build/espeak.zip"
if [[ ! -f $archive ]]; then
    curl -fLsS --retry 2 "https://codeload.github.com/csukuangfj/espeak-ng/zip/$revision" -o "$archive"
fi
printf 'e4e262cbe34f7fe21f91f1ba3397f2728e1f30eafbae7853f2b753a9ed13f0dd  %s\n' "$archive" \
    | sha256sum --check
src="$build/espeak-ng-$revision"
if [[ ! -d $src ]]; then
    unzip -q "$archive" -d "$build"
fi
# Keep the Unicode helper symbols in one Android shared library.
sed -i 's/^add_library(ucd$/add_library(ucd STATIC/' "$src/src/ucd-tools/CMakeLists.txt"
# The fork's blank standalone rpath otherwise consumes the following linker argument.
sed -i '/target_link_libraries(espeak-ng PRIVATE "-Wl,-rpath,\${ESPEAK_NG_RPATH_ORIGIN}")/d' \
    "$src/src/libespeak-ng/CMakeLists.txt"
cmake -G Ninja -S "$src" -B "$build/android" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_ESPEAK_NG_EXE=OFF -DBUILD_ESPEAK_NG_TESTS=OFF \
    -DUSE_MBROLA=OFF -DUSE_LIBSONIC=OFF -DUSE_LIBPCAUDIO=OFF -DUSE_ASYNC=OFF -DUSE_SPEECHPLAYER=OFF
cmake --build "$build/android" --target espeak-ng --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"
cp "$build/android/src/libespeak-ng/libespeak-ng.so" build/stripped/kokoro/
