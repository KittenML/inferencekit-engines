# Run the single-request experiment over USB

Connect a base Galaxy S25 with USB debugging. The APK starts a fresh process for every cell.

```bash
adb install -r build/s25-study.apk
adb shell am start -W -S -n org.inferencekit.s25study/.MainActivity --es plan_name confirmation
```

When the app displays DONE, export the capture.

```bash
mkdir -p results/phone/confirmation
adb exec-out run-as org.inferencekit.s25study cat files/study/latest/bundle.zip \
  > results/phone/confirmation/bundle.zip
unzip results/phone/confirmation/bundle.zip -d results/phone/confirmation
adb exec-out run-as org.inferencekit.s25study cat files/study/latest/index.json \
  > results/phone/confirmation/index.json
```

[Build and analyze](../REPRODUCE.md)
