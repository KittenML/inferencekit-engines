#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/latency"
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to Android NDK r27d}"
: "${ANDROID_SDK_ROOT:?Set ANDROID_SDK_ROOT to Android SDK 35}"
: "${JAVA_HOME:?Set JAVA_HOME to JDK 17}"
export PATH="$JAVA_HOME/bin:$PATH"
export ANDROID_NDK_HOME="$(realpath "$ANDROID_NDK_HOME")"
export ANDROID_SDK_ROOT="$(realpath "$ANDROID_SDK_ROOT")"
export JAVA_HOME="$(realpath "$JAVA_HOME")"
rustup target add aarch64-linux-android
bash scripts/prepare_vieneu.sh
bash pico/prepare.sh all
mkdir -p build/payload/pico
cp -a pico/build/assets/. build/payload/pico/
bash scripts/link_runner.sh
bash scripts/phone-build.sh "$PWD/build/s25-study.apk"
