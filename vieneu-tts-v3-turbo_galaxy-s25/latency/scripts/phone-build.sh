#!/usr/bin/env bash
set -euo pipefail
study=$(cd "$(dirname "$0")/.." && pwd)
sdk=${ANDROID_SDK_ROOT:-}
tools=${PHONE_BUILD_TOOLS:-"$sdk/build-tools/35.0.0"}
platform=${PHONE_ANDROID_JAR:-"$sdk/platforms/android-35/android.jar"}
runner=${PHONE_RUNNER_DIR:-"$study/build/runner"}
payload=${PHONE_PAYLOAD_DIR:-"$study/build/payload"}
plans=${PHONE_PLAN_DIR:-"$study/app/assets"}
output=${1:-"$study/build/s25-study.apk"}
if [[ -n ${JAVA_HOME:-} ]]; then export PATH="$JAVA_HOME/bin:$PATH"; fi

test -s "$runner/libstudy_runner.so"
test -d "$payload"
test -s "$plans/plan.tsv"
mkdir -p "$study/build" "$(dirname "$output")"
build=$(mktemp -d "$study/build/apk.XXXXXXXX")
trap 'rm -r -- "$build"' EXIT
mkdir -p "$build/classes" "$build/dex"
javac --release 8 -encoding UTF-8 -cp "$platform" -d "$build/classes" "$study/app/MainActivity.java"
mapfile -d '' classes < <(find "$build/classes" -type f -name '*.class' -print0)
"$tools/d8" --min-api 28 --lib "$platform" --output "$build/dex" "${classes[@]}"
"$tools/aapt2" link -o "$build/base.apk" -I "$platform" --manifest "$study/app/AndroidManifest.xml"
python3 "$study/scripts/phone-zip.py" \
    "$build/base.apk" "$build/dex/classes.dex" "$runner" "$payload" "$plans"
"$tools/zipalign" -P 16 -f 4 "$build/base.apk" "$build/aligned.apk"
keystore="$study/build/phone-debug.jks"
if [[ ! -f $keystore ]]; then
    keytool -genkeypair -keystore "$keystore" -storepass android -keypass android \
        -alias androiddebugkey -keyalg RSA -keysize 2048 -validity 1000 -dname 'CN=S25 study'
fi
"$tools/apksigner" sign --ks "$keystore" --ks-pass pass:android --v4-signing-enabled false \
    --out "$build/signed.apk" "$build/aligned.apk"
"$tools/apksigner" verify "$build/signed.apk"
"$tools/zipalign" -P 16 -c 4 "$build/signed.apk"
mv "$build/signed.apk" "$output"
sha256sum "$output"
