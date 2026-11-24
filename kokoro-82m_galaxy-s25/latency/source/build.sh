#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=$PWD
kit="${ANDROID_TOOLCHAIN:?Set ANDROID_TOOLCHAIN to a directory containing Android API 35, build-tools, NDK r27d, and JDK 17}"
tools="${ANDROID_BUILD_TOOLS:-$kit/android-15}"
platform="${ANDROID_JAR:-$kit/android-35/android.jar}"
ndk="${ANDROID_NDK:-$kit/android-ndk-r27d}"
export JAVA_HOME="${JAVA_HOME:-$kit/jdk-17.0.21+9}"
export PATH="$JAVA_HOME/bin:$PATH"
mode="${1:-final}"
case "$mode" in final|smoke|compute|acoustic|recovery|ui|acoustic-ui|seeded|matched-chunks|precision-matrix|release-panel) ;; *) echo "Unknown APK mode $mode" >&2; exit 2 ;; esac

mkdir -p build/sherpa-1136/include/sherpa-onnx/c-api build/models build/apk-"$mode"
fetch() {
    local out=$1 url=$2 digest=$3
    if [[ ! -f "$out" ]]; then
        curl --fail --location --silent --show-error --retry 2 --max-time 600 "$url" -o "$out.part"
        mv "$out.part" "$out"
    fi
    printf '%s  %s\n' "$digest" "$out" | sha256sum --check --status
}
runtime_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.13.6/sherpa-onnx-v1.13.6-android.tar.bz2
model_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
fetch build/sherpa-android-1.13.6.tar.bz2 "$runtime_url" bb7f891b259f4faee5c55d4cc10d79a3ba2b27416e9320d2c4d044400834d75b
fetch build/kokoro-v1.0.tar.bz2 "$model_url" 60703707bc05ccd48d60e1eb723f38b07d6e474795a0b6a588d7220400e0f7a4
c_api=build/sherpa-1136/include/sherpa-onnx/c-api/c-api.h
fetch "$c_api" "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/1cb484af5e69d3c7803c1eb0b3b5ab8041e0e911/sherpa-onnx/c-api/c-api.h" '426db2c6acfb51e02143aece67c45779fae699d961c7c26ccf6f1388fdeaa2df'
if [[ ! -f build/sherpa-1136/jniLibs/arm64-v8a/libsherpa-onnx-c-api.so ]]; then
    tar -xjf build/sherpa-android-1.13.6.tar.bz2 -C build/sherpa-1136 ./jniLibs/arm64-v8a
fi
if [[ ! -f build/kokoro-multi-lang-v1_0/voices.bin ]]; then
    tar -xjf build/kokoro-v1.0.tar.bz2 -C build
fi
pico_lib=source/pico/build/libpico_bench.so
if [[ "$mode" != precision-matrix && "$mode" != release-panel && ! -f "$pico_lib" ]]; then
    ANDROID_NDK="$ndk" bash source/pico/build.sh
fi
deps="$root/build/sherpa-1136/jniLibs/arm64-v8a"
apkb="$root/build/apk-$mode"
stage="$apkb/package"
mkdir -p "$apkb/classes" "$stage/assets" "$stage/lib/arm64-v8a"
arm="$stage/lib/arm64-v8a"
compiler="$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang++"
"$compiler" -std=c++17 -O3 -fPIC -shared -Wl,-z,max-page-size=16384 -static-libstdc++ \
    -I build/sherpa-1136/include source/kokoro/kokoro_jni.cpp -L "$deps" \
    -lsherpa-onnx-c-api -Wl,--no-undefined -o "$arm/libkokoro_bench.so"
if [[ "$mode" == precision-matrix || "$mode" == release-panel ]]; then
    pico_lib="$apkb/libpico_bench.so"
    matrix_native="$root/build/matrix-native"
    "$compiler" -std=c++20 -O3 -Wall -Wextra -Werror -fPIC -fvisibility=hidden \
        -static-libstdc++ -fopenmp -static-openmp -Wl,--exclude-libs,ALL -Wl,--gc-sections \
        -Wl,--no-undefined -Wl,-z,max-page-size=16384 -s -shared -DBENCH_PRECISION_MATRIX \
        -I"$matrix_native/include" source/pico/pico_jni.cpp "$matrix_native/lib/libkitten_tts08.a" \
        -ldl -llog -o "$pico_lib"
fi
cp "$deps/"lib{onnxruntime,sherpa-onnx-c-api}.so "$pico_lib" "$arm/"
media_sources=()
media_libraries=()
classpath="$platform"
if [[ "$mode" == release-panel ]]; then
    : "${ADRENO_KOKORO_KIT:?Set ADRENO_KOKORO_KIT to the pinned Android Adreno build}"
    cache="${MEDIAPIPE_CACHE:-$root/build/mediapipe-deps}"
    mkdir -p "$cache"
    while read -r digest repository coordinate; do
        if [[ "$repository" == google ]]; then
            host=https://dl.google.com/dl/android/maven2
        else
            host=https://repo.maven.apache.org/maven2
        fi
        file="$cache/${coordinate##*/}"
        fetch "$file" "$host/$coordinate" "$digest"
        if [[ "$file" == *.aar ]]; then
            jar="$apkb/${file##*/}.jar"
            unzip -p "$file" classes.jar > "$jar"
            while IFS= read -r entry; do
                unzip -p "$file" "$entry" > "$arm/${entry##*/}"
            done < <(unzip -Z1 "$file" 'jni/arm64-v8a/*.so')
            file="$jar"
        fi
        media_libraries+=("$file")
        classpath+=":$file"
    done < source/mediapipe/dependencies.sha256
    media_sources=(source/mediapipe/MediaPipeEngine.java)
    cp "$ADRENO_KOKORO_KIT/Kokoro_82M_inference_fp16" "$arm/libstudy_adreno_worker.so"
    cp "$ADRENO_KOKORO_KIT/libclblast.so" "$arm/"
    "$compiler" -std=c++17 -O3 -Wall -Wextra -Werror -fPIC -shared -static-libstdc++ \
        -Wl,-z,max-page-size=16384 source/kokoro/adreno_jni.cpp -llog -o "$arm/libstudy_adreno.so"
fi
models_zip="$root/build/models.zip"
model_patch=""
if [[ "$mode" == seeded || "$mode" == matched-chunks || "$mode" == precision-matrix ]]; then
    models_zip="$root/build/models-seeded.zip"
    model_patch="$root/build/kokoro-seeded.onnx"
    if [[ ! -f "$model_patch" || source/kokoro/seed_model.py -nt "$model_patch" ]]; then
        uv run --no-project --python 3.11 --with onnx==1.22.0 python source/kokoro/seed_model.py \
            build/kokoro-multi-lang-v1_0/model.onnx "$model_patch" build/kokoro-seeded.json
    fi
fi
if [[ "$mode" == release-panel ]]; then
    models_zip="$root/build/models-release-panel.zip"
    python3 source/pack_release_panel.py "$ADRENO_KOKORO_KIT" "$models_zip.part"
    mv "$models_zip.part" "$models_zip"
elif [[ "$mode" == precision-matrix ]]; then
    models_zip="$root/build/models-precision-matrix.zip"
    python3 - "$models_zip.part" <<'PYCODE'
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

kokoro = Path("build/kokoro-multi-lang-v1_0")
build = Path("build")
with ZipFile(sys.argv[1], "w", compression=ZIP_DEFLATED, compresslevel=1) as out:
    out.write(build / "kokoro-seeded.onnx", "kokoro/model.onnx")
    out.write(build / "kokoro-seeded-int8.onnx", "kokoro/model.int8.onnx")
    files = [kokoro / name for name in ("voices.bin", "tokens.txt", "lexicon-us-en.txt")]
    files.extend(p for p in (kokoro / "espeak-ng-data").rglob("*") if p.is_file())
    for file in sorted(files):
        out.write(file, f"kokoro/{file.relative_to(kokoro).as_posix()}")
    for model in ("mini", "pico"):
        base = build / "matrix-models" / model
        prefix = "kitten_tts_2_mini" if model == "mini" else "kitten_tts_2_pico_en"
        files = [base / name for name in ("vocab.json", "eleanor.bin")]
        for precision in ("fp32", "int8", "int4"):
            files.extend([base / f"{prefix}_{precision}.bin", base / f"{prefix}_{precision}_arch.json"])
        files.extend(p for p in (base / "g2p").rglob("*") if p.is_file())
        for file in sorted(files):
            out.write(file, f"{model}/{file.relative_to(base).as_posix()}")
PYCODE
    mv "$models_zip.part" "$models_zip"
elif [[ ! -f "$models_zip" || source/pico/build/package-sha256.txt -nt "$models_zip" ||
      ( -n "$model_patch" && "$model_patch" -nt "$models_zip" ) ]]; then
    python3 - "$models_zip.part" "$model_patch" <<'PYCODE'
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

archive = Path("build/kokoro-multi-lang-v1_0")
pico = Path("source/pico/assets/models/pico")
kokoro_files = [archive / x for x in ("model.onnx", "voices.bin", "tokens.txt", "lexicon-us-en.txt")]
kokoro_files.extend(p for p in (archive / "espeak-ng-data").rglob("*") if p.is_file())
with ZipFile(sys.argv[1], "w", compression=ZIP_DEFLATED, compresslevel=3) as out:
    for label, base, files in (("kokoro", archive, kokoro_files),
                               ("pico", pico, [p for p in pico.rglob("*") if p.is_file()])):
        for file in sorted(files):
            payload = sys.argv[2] if file == archive / "model.onnx" and sys.argv[2] else file
            out.write(payload, f"{label}/{file.relative_to(base).as_posix()}")
PYCODE
    mv "$models_zip.part" "$models_zip"
fi
cp fixtures/*.jsonl "$stage/assets/"
ln -sf "$models_zip" "$stage/assets/models.zip"
printf '%s\n' "$mode" > "$stage/assets/mode.txt"
javac --release 8 -classpath "$classpath" -d "$apkb/classes" source/android/*.java \
    source/acoustic/AcousticCapture.java "${media_sources[@]}"
mapfile -d '' classes < <(find "$apkb/classes" -name '*.class' -print0)
"$tools/d8" --min-api 31 --lib "$platform" --output "$stage" "${classes[@]}" "${media_libraries[@]}"
manifest=source/android/AndroidManifest.xml
if [[ "$mode" == acoustic || "$mode" == acoustic-ui || "$mode" == seeded ]]; then
    manifest="$apkb/AndroidManifest.xml"
    sed '/<uses-sdk/a\  <uses-permission android:name="android.permission.RECORD_AUDIO" />' \
        source/android/AndroidManifest.xml > "$manifest"
fi
if [[ "$mode" == release-panel ]]; then
    manifest="$apkb/AndroidManifest.xml"
    sed '/<activity android:name=/i\    <uses-native-library android:name="libOpenCL.so" android:required="false" />' \
        source/android/AndroidManifest.xml > "$manifest"
fi
"$tools/aapt2" link -o "$apkb/unsigned.apk" -I "$platform" --manifest "$manifest"
(cd "$stage" && zip -qr -0 "$apkb/unsigned.apk" classes*.dex assets lib)
"$tools/zipalign" -f -P 16 4 "$apkb/unsigned.apk" "$apkb/aligned.apk"
if [[ ! -f build/study-debug.jks ]]; then
    keytool -genkeypair -keystore build/study-debug.jks -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 1000 \
        -dname 'CN=Galaxy S25 local benchmark' >/dev/null
fi
apk="$root/build/s25-tts-$mode.apk"
"$tools/apksigner" sign --ks build/study-debug.jks --ks-pass pass:android --out "$apk" "$apkb/aligned.apk"
"$tools/apksigner" verify "$apk"
printf '%s\n' "$apk"
sha256sum "$apk"
