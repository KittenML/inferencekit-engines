#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
if [[ -n ${ANDROID_SDK_ROOT:-} ]]; then
    tools="$ANDROID_SDK_ROOT/build-tools/35.0.0"
    platform="$ANDROID_SDK_ROOT/platforms/android-35/android.jar"
else
    tools="$root/toolchain/android-15"
    platform="$root/toolchain/android-35/android.jar"
fi
if [[ -z ${JAVA_HOME:-} && -d $root/toolchain ]]; then
    JAVA_HOME=$(find "$root/toolchain" -maxdepth 1 -name 'jdk-*' -print -quit)
fi
if [[ -n ${JAVA_HOME:-} ]]; then
    export JAVA_HOME PATH="$JAVA_HOME/bin:$PATH"
fi
mkdir -p build/apk/classes build/apk/dex
javac -source 8 -target 8 -Xlint:-options -classpath "$platform" -d build/apk/classes source/app/MainActivity.java
"$tools/d8" --min-api 28 --lib "$platform" --output build/apk/dex build/apk/classes/org/inferencekit/freshcapture/MainActivity.class
"$tools/aapt2" link -o build/apk/unsigned.apk -I "$platform" --manifest source/app/AndroidManifest.xml
zip -qj build/apk/unsigned.apk build/apk/dex/classes.dex
"$tools/zipalign" -f -p 4 build/apk/unsigned.apk build/apk/aligned.apk
if [[ ! -f build/apk/debug.jks ]]; then
    keytool -genkeypair -keystore build/apk/debug.jks -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 1000 \
        -dname 'CN=InferenceKit Benchmark' >/dev/null 2>&1
fi
"$tools/apksigner" sign --ks build/apk/debug.jks --ks-pass pass:android --out build/capture.apk build/apk/aligned.apk
"$tools/apksigner" verify build/capture.apk
