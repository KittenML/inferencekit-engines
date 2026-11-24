#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
source_dir="$root/source/mediapipe"
android="$root/toolchain/android-35/android.jar"
tools="$root/toolchain/android-15"
if [[ -n ${ANDROID_SDK_ROOT:-} ]]; then
    android="$ANDROID_SDK_ROOT/platforms/android-35/android.jar"
    tools="$ANDROID_SDK_ROOT/build-tools/35.0.0"
fi
if [[ -z ${JAVA_HOME:-} && -d $root/toolchain ]]; then
    JAVA_HOME=$(find "$root/toolchain" -maxdepth 1 -name 'jdk-*' -print -quit)
fi
if [[ -n ${JAVA_HOME:-} ]]; then
    export JAVA_HOME PATH="$JAVA_HOME/bin:$PATH"
fi

version=${1:-1.2.2}
cache=${INFERENCEKIT_MEDIAPIPE_CACHE:-"$root/build/mediapipe/dependencies"}
build="$root/build/mediapipe"
apk="$root/build/mediapipe-capture.apk"
manifest_args=()
dependencies() {
    if [[ $version == 1.2.3 ]]; then
        cat "$source_dir/dependencies-1.2.3.sha256"
        tail -n +3 "$source_dir/dependencies.sha256"
    else
        cat "$source_dir/dependencies.sha256"
    fi
}
case "$version" in
    1.2.2) ;;
    1.2.3)
        build="$root/build/mediapipe-1.2.3/build"
        cache="$root/build/mediapipe-1.2.3"
        apk="$root/build/mediapipe-capture-123.apk"
        app=org.inferencekit.mediapipecapture.v123
        manifest_args=(--rename-manifest-package "$app" --rename-instrumentation-target-package "$app")
        ;;
    *) echo 'Choose pinned MediaPipe Tasks Audio 1.2.2 or 1.2.3.' >&2; exit 2 ;;
esac
mkdir -p "$cache" "$build"/{class,apk/lib/arm64-v8a}
libraries=()
while read -r digest repository coordinate; do
    if [[ $repository == google ]]; then
        host=https://dl.google.com/dl/android/maven2
    else
        host=https://repo.maven.apache.org/maven2
    fi
    file="$cache/${coordinate##*/}"
    if [[ ! -f $file ]]; then
        curl -fLsS --retry 2 "$host/$coordinate" -o "$file.partial"
        mv "$file.partial" "$file"
    fi
    printf '%s  %s\n' "$digest" "$file" | sha256sum --check --status
    if [[ $file == *.aar ]]; then
        jar="$build/${file##*/}.jar"
        unzip -p "$file" classes.jar > "$jar"
        while IFS= read -r entry; do
            unzip -p "$file" "$entry" > "$build/apk/lib/arm64-v8a/${entry##*/}"
        done < <(unzip -Z1 "$file" 'jni/arm64-v8a/*.so')
        libraries+=("$jar")
    else
        libraries+=("$file")
    fi
done < <(dependencies)

classpath=$(IFS=:; printf '%s' "${libraries[*]}")
printf 'package org.inferencekit.mediapipecapture; final class RuntimeVersion { static final String NAME = "%s"; }\n' \
    "$version" > "$build/RuntimeVersion.java"
javac -source 8 -target 8 -Xlint:-options -classpath "$android:$classpath" \
    -d "$build/class" "$source_dir/"*.java "$build/RuntimeVersion.java"
mapfile -d '' classes < <(find "$build/class" -type f -name '*.class' -print0)
"$tools/d8" --min-api 28 --lib "$android" --output "$build/apk" "${classes[@]}" "${libraries[@]}"
"$tools/aapt2" link -o "$build/unsigned.apk" -I "$android" --manifest "$source_dir/AndroidManifest.xml" \
    "${manifest_args[@]}"
(cd "$build/apk" && zip -q -r -0 "$build/unsigned.apk" classes*.dex lib)
"$tools/zipalign" -f 4 "$build/unsigned.apk" "$build/aligned.apk"
if [[ ! -f build/apk/debug.jks ]]; then
    mkdir -p build/apk
    keytool -genkeypair -keystore build/apk/debug.jks -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 1000 \
        -dname 'CN=InferenceKit Benchmark' >/dev/null 2>&1
fi
"$tools/apksigner" sign --ks build/apk/debug.jks --ks-pass pass:android \
    --out "$apk" "$build/aligned.apk"
"$tools/apksigner" verify "$apk"
