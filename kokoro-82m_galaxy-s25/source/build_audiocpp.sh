#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
ndk=${ANDROID_NDK_HOME:-"$root/toolchain/android-ndk-r27d"}
build=build-audio-android
source_tree=upstream-audiocpp
openmp=OFF
model_family=vieneu_v3_turbo
variant=${1:-stock}
options=()
checkout_header() {
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
case "$variant" in
    stock) ;;
    backbone-hidden|kokoro-im2col)
        build="build-audio-android-$variant"
        source_tree="$root/upstream-audiocpp-$variant"
        if [[ $variant == backbone-hidden ]]; then
            patch="$root/source/patches/vieneu-backbone-hidden.patch"
        else
            patch="$root/source/patches/kokoro-fixed-im2col.patch"
            model_family=kokoro_tts
            options=(-DENGINE_ENABLE_NEON=ON)
        fi
        if [[ ! -d $source_tree ]]; then
            git -C upstream-audiocpp worktree add --detach "$source_tree" 749fffec6ad4a763520388a4240cd24e08a0a2bf
        fi
        if git -C "$source_tree" apply --check "$patch" 2>/dev/null; then
            git -C "$source_tree" apply "$patch"
        else
            git -C "$source_tree" apply --reverse --check "$patch"
        fi
        ;;
    kleidiai)
        build=build-audio-android-kleidiai
        options=(-DGGML_CPU_KLEIDIAI=ON)
        ;;
    kokoro)
        build=build-audio-android-kokoro
        model_family=kokoro_tts
        options=(-DENGINE_ENABLE_NEON=ON)
        ;;
    openmp)
        build=build-audio-android-openmp
        openmp=ON
        libomp="$ndk/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/lib/linux/aarch64/libomp.a"
        options=(
            -DOpenMP_C_FLAGS=-fopenmp=libomp -DOpenMP_CXX_FLAGS=-fopenmp=libomp
            -DOpenMP_C_LIB_NAMES=omp -DOpenMP_CXX_LIB_NAMES=omp "-DOpenMP_omp_LIBRARY=$libomp"
        )
        ;;
    vulkan|kokoro-vulkan)
        build="build-audio-android-$variant"
        if [[ $variant == kokoro-vulkan ]]; then
            model_family=kokoro_tts
        fi
        checkout_header https://github.com/KhronosGroup/Vulkan-Headers.git upstream-vulkan-headers 29184b98984f6169a5e83e97557a77cff1e5b0ca
        checkout_header https://github.com/KhronosGroup/SPIRV-Headers.git upstream-spirv-headers 04f10f650d514df88b76d25e83db360142c7b174
        ndk_library="$ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/28/libvulkan.so"
        options=(
            -DENGINE_ENABLE_VULKAN=ON "-DVulkan_INCLUDE_DIR=$root/upstream-vulkan-headers/include"
            "-DVulkan_LIBRARY=$ndk_library" "-DVulkan_GLSLC_EXECUTABLE=$ndk/shader-tools/linux-x86_64/glslc"
        )
        export CPATH="$root/upstream-spirv-headers/include${CPATH:+:$CPATH}"
        ;;
    *) echo 'Select stock, backbone-hidden, kleidiai, kokoro, kokoro-im2col, kokoro-vulkan, openmp or vulkan.' >&2; exit 1 ;;
esac
cmake -G Ninja -S "$source_tree" -B "$build" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_MODEL_SET=custom "-DAUDIOCPP_MODELS=$model_family" \
    -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DAUDIOCPP_BUILD_C_API=ON -DENGINE_ENABLE_NATIVE_CPU=OFF \
    -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm -DENGINE_ENABLE_LLAMAFILE=OFF \
    -DENGINE_ENABLE_OPENMP="$openmp" -DGGML_OPENMP="$openmp" "${options[@]}"
cmake --build "$build" --target audiocpp --parallel "${INFERENCEKIT_BUILD_JOBS:-4}"
if [[ $variant != stock ]]; then
    bin="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin"
    mkdir -p "build/stripped/$variant"
    "$bin/llvm-strip" --strip-all -o "build/stripped/$variant/libaudiocpp.so" "$build/bin/libaudiocpp.so"
fi
