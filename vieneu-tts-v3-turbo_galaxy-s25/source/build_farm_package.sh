#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
phase=${1:-screen2}
if [[ ! -f build/stripped/capture ]]; then
    ./install.sh baseline
fi
python3 -m venv py
py/bin/python -m pip install -r source/farm-requirements.txt
mkdir -p build/wheelhouse
py/bin/python -m pip download --dest build/wheelhouse --no-deps \
    pytest==7.4.4 pluggy==1.2.0 packaging==24.0 iniconfig==2.0.0 \
    exceptiongroup==1.2.2 tomli==2.0.1 typing_extensions==4.12.2 importlib_metadata==6.7.0 zipp==3.15.0
bash source/build_apk.sh
mapfile -t variants < <(python3 -c '
import json
import sys
with open(sys.argv[1]) as plan:
    arms = json.load(plan)["arms"]
variants = {arm[5].get("INFERENCEKIT_AUDIORUNTIME", "stock") for arm in arms if len(arm) > 5}
models = {arm[1] for arm in arms}
if any(model.startswith("kokoro-mediapipe-") and not model.endswith("-123") for model in models):
    variants.add("mediapipe")
if any(model.startswith("kokoro-mediapipe-") and model.endswith("-123") for model in models):
    variants.add("mediapipe-123")
if "kokoro-audiocpp-fp32" in models:
    variants.add("kokoro")
if "kokoro-adreno-fp16" in models:
    variants.add("adreno-kokoro")
if any(model.startswith("kokoro-soniqo-") for model in models):
    variants.add("soniqo-kokoro")
if "vieneu-cpp" in models or any(model.startswith("vieneu-cpp-snap-") for model in models):
    variants.add("vieneu-cpp")
if "vieneu-cpp-snap-mixed" in models:
    variants.add("vieneu-cpp-snapdragon-mixed")
elif any(model.startswith("vieneu-cpp-snap-") for model in models):
    variants.add("vieneu-cpp-snapdragon")
print("\n".join(sorted(variants - {"stock"})))
' "source/$phase.json")
for variant in "${variants[@]}"; do
    case "$variant" in
        '') ;;
        mediapipe) bash source/build_mediapipe_apk.sh ;;
        mediapipe-123) bash source/build_mediapipe_apk.sh 1.2.3 ;;
        adreno-kokoro) bash source/build_adreno_kokoro.sh ;;
        soniqo-kokoro)
            if [[ -z ${INFERENCEKIT_SONIQO_KOKORO_KIT:-} && ! -f build/soniqo-kokoro/android-kit/capture-soniqo-kokoro ]]; then
                root="$PWD"
                mkdir -p build/soniqo-kokoro
                (
                    cd build/soniqo-kokoro
                    bash "$root/source/competitors/soniqo-kokoro/reproduce.sh" \
                        "${ANDROID_NDK_HOME:-$root/toolchain/android-ndk-r27d}" "$root/source/capture.cpp"
                )
            fi
            ;;
        vieneu-cpp)
            if [[ -z ${INFERENCEKIT_VIENEU_CPP_KIT:-} ]]; then
                if [[ ! -f build/vieneu-cpp/android-kit/capture-vieneu-cpp ]]; then
                    ./install.sh vieneu-cpp
                fi
                export INFERENCEKIT_VIENEU_CPP_KIT="$PWD/build/vieneu-cpp/android-kit"
            fi
            ;;
        vieneu-cpp-snapdragon|vieneu-cpp-snapdragon-mixed)
            if [[ $variant == vieneu-cpp-snapdragon-mixed ]]; then
                placement=mixed
                layout=android-snapdragon-mixed
                kit_variable=INFERENCEKIT_VIENEU_SNAPDRAGON_MIXED_KIT
            else
                placement=scalar
                layout=android-snapdragon
                kit_variable=INFERENCEKIT_VIENEU_SNAPDRAGON_KIT
            fi
            if [[ -z ${!kit_variable:-} ]]; then
                root="$PWD"
                if [[ ! -f build/vieneu-cpp/$layout/capture-vieneu-cpp ]]; then
                    if [[ ! -f build/vieneu-cpp/android-kit/capture-vieneu-cpp ]]; then
                        ./install.sh vieneu-cpp
                    fi
                    (
                        cd build/vieneu-cpp
                        bash "$root/source/competitors/vieneu-tts-cpp/reproduce-snapdragon.sh" \
                            "${ANDROID_NDK_HOME:-$root/toolchain/android-ndk-r27d}" "$placement"
                    )
                fi
                export "$kit_variable=$root/build/vieneu-cpp/$layout"
            fi
            ;;
        ort-pool)
            if [[ ! -f build/stripped/ort-pool/libsherpa-onnx-c-api.so ]]; then
                bash source/build_kokoro_ort_pool.sh
            fi
            ;;
        vieneu-shared) test -f build/stripped/vieneu-shared/capture ;;
        vieneu-cpp-shared) test -f build/stripped/vieneu-cpp-shared/libvieneu-tts.so ;;
        vieneu-cpp-batched) test -f build/stripped/vieneu-cpp-batched/libvieneu-tts.so ;;
        kokoro)
            bash source/build_audiocpp.sh kokoro
            bash source/build_espeak_android.sh
            ;;
        *) bash source/build_audiocpp.sh "$variant" ;;
    esac
done
python3 source/package.py "source/$phase.json" "build/$phase.zip"
