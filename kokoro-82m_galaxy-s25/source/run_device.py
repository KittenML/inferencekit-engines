"""Runs signed-release benchmarking cells serially on a Device Farm Android host."""
import argparse
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
import wave

from prepare_payload import digest_of, prepare

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path(os.environ.get("DEVICEFARM_LOG_DIR", ".")) / "speech-capture"
DEVICE = "/data/local/tmp/inferencekit-fresh-7s3a5q7o"
LABEL = re.compile(r"[a-z0-9][a-z0-9-]{0,62}")
PLAN_FILE = re.compile(r"[a-z0-9][a-z0-9-]{0,62}\.json")
MEDIAPIPE_APP = "org.inferencekit.mediapipecapture"
MEDIAPIPE_MODELS = {
    "kokoro-mediapipe-cpu": ("CPU", "CPU"),
    "kokoro-mediapipe-gpu": ("GPU", "GPU"),
    "kokoro-mediapipe-gpu-bundle-on-cpu": ("CPU", "GPU"),
    "kokoro-mediapipe-gpu-123": ("GPU", "GPU"),
    "kokoro-mediapipe-npu-123": ("NPU", "GPU"),
}
SNAPDRAGON_MODELS = {"vieneu-cpp-snap-cpu": "", "vieneu-cpp-snap-htp": "HTP0",
                    "vieneu-cpp-snap-opencl": "GPUOpenCL", "vieneu-cpp-snap-mixed": None}

MODELS = {
    "kokoro-fp32": "models/kokoro-multi-lang-v1_0/model.onnx",
    "kokoro-int8": "models/kokoro-int8-multi-lang-v1_0/model.int8.onnx",
    "kokoro-audiocpp-fp32": "models/kokoro-audiocpp/kokoro-82m-f32.gguf",
    "kokoro-adreno-fp16": "kokoro-adreno",
    "kokoro-litert-omni-gpu": "models/mediapipe/kokoro-GPU.litertlm",
    "kokoro-soniqo-cpu": "soniqo-kokoro/model",
    "kokoro-soniqo-nnapi": "soniqo-kokoro/model",
    "vieneu-bf16": "models/vieneu/vieneu-v3-turbo-bf16.gguf",
    "vieneu-q8": "models/vieneu/vieneu-v3-turbo-q8_0.gguf",
    "vieneu-cpp": "vieneu-cpp/model",
    **dict.fromkeys(SNAPDRAGON_MODELS, "vieneu-cpp/model"),
}


def adb(*args, timeout=200, check=True):
    return subprocess.run(("adb", *args), check=check, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def adb_metadata(name, *commands):
    with (OUTPUT / name).open("wb") as stream:
        for command in commands:
            result = adb("shell", command, check=False, timeout=15)
            stream.write(f"$ {command}\n".encode())
            stream.write(result.stdout)
            stream.write(result.stderr)


def condition_thermal(cell, config):
    started = time.monotonic()
    target = config["max_status"]
    readings = []
    while True:
        result = adb("shell", "dumpsys thermalservice", check=False, timeout=12)
        observed = re.search(rb"Thermal Status: (\d+)", result.stdout)
        status = int(observed[1]) if observed else None
        seconds = time.monotonic() - started
        readings.append({"elapsed_s": round(seconds, 2), "status": status})
        reached = status is not None and status <= target
        if reached or status is None or seconds >= config["timeout_s"]:
            break
        time.sleep(min(config["interval_s"], max(0, config["timeout_s"] - seconds)))
    (OUTPUT / f"{cell}-thermal-precondition.json").write_text(json.dumps({
        "target_max_status": target, "reached": reached, "readings": readings,
    }, indent=2) + "\n")
    print(f"{cell} thermal start target={target} observed={status} waited_s={seconds:.1f}", flush=True)


def selected_binary(model, runtime):
    if model == "kokoro-litert-omni-gpu":
        return ("./omni-kokoro/capture-omni-kokoro", "export LD_LIBRARY_PATH=$PWD/omni-kokoro/lib")
    if model.startswith("kokoro-soniqo-"):
        request_nnapi = "1" if model.endswith("-nnapi") else "0"
        return ("./soniqo-kokoro/capture-soniqo-kokoro",
                f"export LD_LIBRARY_PATH=$PWD/soniqo-kokoro/lib SONIQO_KOKORO_NNAPI={request_nnapi}")
    if model == "kokoro-adreno-fp16":
        return (
            "./bin/iktts7s3a5q7o",
            "export LD_LIBRARY_PATH=$PWD/kokoro-adreno:$PWD/lib "
            "KOKORO_ADRENO=1 NNOPT_GPU_FP32_GENERATOR=1",
        )
    if model in SNAPDRAGON_MODELS:
        device = SNAPDRAGON_MODELS[model]
        target = ""
        if device is not None:
            target = f"VIENEU_BACKBONE_DEVICE={device}" if device else "GGML_HEXAGON_NDEV=0"
        return (
            "./vieneu-snapdragon/capture-vieneu-cpp",
            "export LD_LIBRARY_PATH=$PWD/vieneu-snapdragon/lib "
            "ADSP_LIBRARY_PATH=$PWD/vieneu-snapdragon/lib "
            f"VIENEU_SEA_G2P_DICT=$PWD/vieneu-cpp/sea_g2p.bin {target}",
        )
    if model == "vieneu-cpp":
        libraries = "$PWD/vieneu-cpp/lib"
        if runtime in {"vieneu-cpp-shared", "vieneu-cpp-batched"}:
            libraries = f"$PWD/vieneu-cpp/lib/{runtime.removeprefix('vieneu-cpp-')}:" + libraries
        return (
            "./vieneu-cpp/capture-vieneu-cpp",
            f"export LD_LIBRARY_PATH={libraries} "
            "VIENEU_SEA_G2P_DICT=$PWD/vieneu-cpp/sea_g2p.bin",
        )
    if model == "kokoro-audiocpp-fp32":
        libraries = "$PWD/lib/kokoro:$PWD/lib"
        if runtime in {"kokoro-im2col", "kokoro-vulkan"}:
            libraries = f"$PWD/lib/{runtime}:" + libraries
        return (
            "./bin/iktts7s3a5q7o",
            f"export LD_LIBRARY_PATH={libraries} KOKORO_AUDIOCPP=1 "
            "AUDIOCPP_ESPEAK_LIBRARY=$PWD/lib/kokoro/libespeak-ng.so "
            "AUDIOCPP_ESPEAK_DATA=$PWD/models/kokoro-multi-lang-v1_0/espeak-ng-data",
        )
    libraries = "$PWD/lib" if runtime == "stock" else f"$PWD/lib/{runtime}:$PWD/lib"
    if runtime == "vieneu-shared":
        return "./lib/vieneu-shared/capture", f"export LD_LIBRARY_PATH={libraries}"
    return "./bin/iktts7s3a5q7o", f"export LD_LIBRARY_PATH={libraries}"


def media_app(model):
    return MEDIAPIPE_APP + ".v123" if model.endswith("-123") else MEDIAPIPE_APP


def stage_mediapipe(payload, models):
    app = media_app(models[0])
    apk = "mediapipe-capture.apk" if app == MEDIAPIPE_APP else "mediapipe-capture-123.apk"
    adb("install", "-r", str(payload / "application" / apk), timeout=120)
    adb("shell", f"run-as {app} mkdir -p files")
    files = {"plans/kokoro.tsv": "kokoro.tsv"}
    for model in models:
        _, bundle = MEDIAPIPE_MODELS[model]
        filename = f"kokoro-{bundle}.litertlm"
        files[f"models/mediapipe/{filename}"] = filename
    for source, destination in files.items():
        write = shlex.quote(f"cat > files/{destination}")
        adb("shell", f"cat {DEVICE}/{source} | run-as {app} sh -c {write}", timeout=120)


def capture_mediapipe(model, round_number, concurrency, waves, period_ms, mode="stream",
                      pool_size=0, pool_policy="deadline", save_pcm="0", foreground="0", target_bucket="64",
                      timeout=0):
    app = media_app(model)
    adb("shell", f"run-as {app} rm -rf files/pcm files/capture.jsonl")
    delegate, bundle = MEDIAPIPE_MODELS[model]
    command = (
        "am", "instrument", "-w", "-e", "delegate", delegate, "-e", "bundle", bundle,
        "-e", "round", str(round_number), "-e", "concurrency", str(concurrency),
        "-e", "waves", str(waves), "-e", "period_ms", str(period_ms), "-e", "mode", mode,
        "-e", "pool_size", str(pool_size), "-e", "pool_policy", pool_policy,
        "-e", "save_pcm", save_pcm, "-e", "foreground", foreground, "-e", "target_bucket", target_bucket,
        f"{app}/{MEDIAPIPE_APP}.MediaPipeCapture",
    )
    return adb("shell", shlex.join(command), timeout=max(timeout, 120, 90 + waves * period_ms / 1000), check=False)


def media_runtime_log(name, instrumentation, output=None):
    pid = re.search(rb"INSTRUMENTATION_RESULT: pid=(\d+)", instrumentation)
    recorded = b""
    if pid:
        log = (output if output is not None else OUTPUT) / f"{name}-delegate.log"
        try:
            selected = adb("logcat", "-d", "--pid=" + pid[1].decode(),
                           "InferenceKitTTS:I", "litert:I", "native:I", "tflite:I", "AdrenoUtils:I", "*:S",
                           check=False, timeout=20)
            recorded = selected.stdout + selected.stderr
        except subprocess.TimeoutExpired:
            recorded = b"delegate log collection timed out\n"
        log.write_bytes(recorded)
    return recorded


def run_mediapipe_cell(label, model, plan, round_number, concurrency, cell, options):
    completed = capture_mediapipe(
        model, round_number, concurrency, plan["waves"], plan["period_ms"], plan["mode"],
        int(options.get("INFERENCEKIT_MEDIAPIPE_POOL_SIZE", 0)),
        options.get("INFERENCEKIT_MEDIAPIPE_POOL_POLICY", "deadline"),
        options.get("INFERENCEKIT_SAVE_PCM", "0"),
        options.get("INFERENCEKIT_MEDIAPIPE_FOREGROUND", "0"),
        options.get("INFERENCEKIT_MEDIAPIPE_BUCKET", "64"),
        int(options.get("INFERENCEKIT_CELL_TIMEOUT", 0)),
    )
    with (OUTPUT / f"{cell}.log").open("wb") as log:
        log.write(completed.stdout)
        log.write(completed.stderr)
    media_runtime_log(cell, completed.stdout)
    if completed.returncode or b"status=complete" not in completed.stdout:
        return 1
    app = media_app(model)
    captured = adb("exec-out", f"run-as {app} cat files/capture.jsonl")
    with (OUTPUT / f"{label}.jsonl").open("ab") as timings:
        timings.write(captured.stdout)
    destination = shlex.quote(f"{DEVICE}/pcm/{label}")
    adb("shell", f"mkdir -p {destination} && run-as {app} tar -cf - -C files/pcm . | "
                 f"tar -xf - -C {destination}")
    return 0


def run_cell(arm, plan, round_number, concurrency):
    label, model, mode, threads, pin, *options = arm
    if not LABEL.fullmatch(label):
        raise ValueError("invalid capture label")
    family = model.split("-")[0]
    cell = f"{label}-r{round_number}-n{concurrency}"
    print(f"{cell} start", flush=True)
    adb_metadata(f"{cell}-before.txt", "dumpsys thermalservice", "cat /proc/meminfo | head -4")
    if model in MEDIAPIPE_MODELS:
        status = run_mediapipe_cell(label, model, plan, round_number, concurrency, cell,
                                    options[0] if options else {})
    else:
        status = run_native_cell(arm, plan, round_number, concurrency, cell, family)
    adb_metadata(f"{cell}-after.txt", "dumpsys thermalservice", "cat /proc/meminfo | head -4")
    if status:
        print(f"{cell} exited {status}", flush=True)
    else:
        print(f"{cell} finished", flush=True)
    return status


def run_native_cell(arm, plan, round_number, concurrency, cell, family):
    label, model, mode, threads, pin, *options = arm
    env = dict(options[0]) if options else {}
    env.pop("INFERENCEKIT_PLAN", None)
    timeout = int(env.pop("INFERENCEKIT_CELL_TIMEOUT", 0))
    runtime = env.pop("INFERENCEKIT_AUDIORUNTIME", "stock")
    if runtime not in {"stock", "backbone-hidden", "kleidiai", "kokoro-im2col", "kokoro-vulkan", "openmp",
                       "ort-pool", "vieneu-cpp-batched", "vieneu-cpp-shared", "vieneu-shared", "vulkan"}:
        raise ValueError(f"{cell}: unknown audio.cpp runtime")
    binary, setup = selected_binary(model, runtime)
    arguments = (
        binary, family, MODELS[model], f"plans/{family}.tsv",
        mode, str(threads), str(pin), str(round_number), str(concurrency),
        str(plan["waves"]), str(plan["period_ms"]), f"pcm/{label}",
    )
    allowed = {"INFERENCEKIT_MEMORY_ROLLUP", "INFERENCEKIT_OMNI_PROCESS_PER_STREAM", "INFERENCEKIT_PROFILE",
               "INFERENCEKIT_SAVE_PCM", "INFERENCEKIT_SENTENCE_WORKERS",
               "KOKORO_AUDIOCPP_BACKEND", "KOKORO_AUDIOCPP_NEON",
               "KOKORO_FP32_FIXED_IM2COL", "KOKORO_ORT_GLOBAL_INTRA_OP", "KOKORO_PROVIDER",
               "KOKORO_SHARED_SESSION", "NNOPT_DOT8", "NNOPT_DOT8_GEMM", "OMP_WAIT_POLICY",
               "KMP_BLOCKTIME", "VIENEU_BACKEND",
               "VIENEU_BACKBONE_BATCH_THREADS", "VIENEU_BACKBONE_DEVICES", "VIENEU_BACKBONE_THREADS",
               "VIENEU_SHARE_CPU_BACKBONE", "VIENEU_BACKBONE_LOG_BUFFERS", "VIENEU_SHARE_CPU_Q8_WEIGHTS",
               "VIENEU_V3_CPU_DISPATCH", "VIENEU_V3_NATIVE_BENCHMARK", "SPEECH_CORE_ORT_PROFILE_PREFIX"}
    if set(env) - allowed:
        raise ValueError(f"{cell}: unknown native runtime option")
    settings = [f"{name}={value}" for name, value in sorted(env.items())]
    executable = ["env", *settings, *arguments] if settings else arguments
    remote = f"cd {DEVICE} && {setup} && mkdir -p {shlex.quote('pcm/' + label)} && " + shlex.join(executable)
    profile = env.get("VIENEU_V3_NATIVE_BENCHMARK") == "1"
    output_name = f"{cell}-profile.txt" if profile else f"{label}.jsonl"
    with (OUTPUT / output_name).open("ab") as timings, (OUTPUT / f"{cell}.log").open("wb") as log:
        log.write((f"audio_cpp_runtime={runtime} " + shlex.join(executable) + "\n").encode())
        log.flush()
        completed = subprocess.run(("adb", "shell", remote), stdout=timings, stderr=log,
                                   timeout=max(timeout, 90, plan["waves"] * plan["period_ms"] / 1000 + 65))
    return completed.returncode


def verify_local_package(payload):
    root = payload.parent / "local-reproduction"
    build = root / "build"
    (build / "vieneu-cpp").mkdir(parents=True, exist_ok=True)
    (build / "vieneu-cpp/android-kit").symlink_to(payload / "vieneu-cpp", target_is_directory=True)
    (build / "mediapipe-capture.apk").symlink_to(payload / "application/mediapipe-capture.apk")
    records = []
    with (OUTPUT / "local-reproduction.log").open("wb") as log:
        for script in ("serve.sh", "test.sh"):
            try:
                call = subprocess.run(("bash", str(root / script)), cwd=root, stdout=log, stderr=log,
                                      timeout=220, check=False)
                records.append({"script": script, "exit_code": call.returncode})
                if call.returncode:
                    break
            except subprocess.TimeoutExpired:
                records.append({"script": script, "timeout_sec": 220})
                break
    (OUTPUT / "local-reproduction-commands.json").write_text(json.dumps(records, indent=2) + "\n")
    samples = []
    for path in sorted((build / "output").glob("*.wav")):
        shutil.copyfile(path, OUTPUT / path.name)
        with wave.open(str(path), "rb") as recording:
            frames, rate, channels = recording.getnframes(), recording.getframerate(), recording.getnchannels()
        samples.append({"profile": path.stem, "sha256": digest_of(path), "frames": frames,
                        "rate": rate, "channels": channels})
    (OUTPUT / "local-reproduction-outputs.json").write_text(json.dumps(samples, indent=2) + "\n")
    print("local engine scripts " + ("PASSED" if len(samples) == 2 else "FAILED"), flush=True)


def smoke_models(arms, payload):
    requested = sorted({arm[1] for arm in arms})
    available = set()
    mediapipe = [model for model in requested if model in MEDIAPIPE_MODELS]
    ready_apps = set()
    for app in sorted({media_app(model) for model in mediapipe}):
        selected = [model for model in mediapipe if media_app(model) == app]
        try:
            stage_mediapipe(payload, selected)
            ready_apps.add(app)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            (OUTPUT / f"kokoro-mediapipe-{app.rsplit('.', 1)[-1]}-setup.log").write_text(str(error))
    for model in requested:
        if model in MEDIAPIPE_MODELS:
            app = media_app(model)
            if app in ready_apps:
                try:
                    result = capture_mediapipe(model, 0, 1, 1, 3000)
                    (OUTPUT / f"{model}-smoke.log").write_bytes(result.stdout + result.stderr)
                    media_runtime_log(f"{model}-smoke", result.stdout)
                    if not result.returncode and b"status=complete" in result.stdout:
                        timings = adb("exec-out", f"run-as {app} cat files/capture.jsonl")
                        if json.loads(timings.stdout).get("complete_ms") is not None:
                            available.add(model)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                    (OUTPUT / f"{model}-smoke.log").write_text(str(error))
            print(f"{model} smoke {'completed' if model in available else 'failed'}", flush=True)
            continue
        family = model.split("-")[0]
        text = "The harbor is quiet." if family == "kokoro" else "Bến cảng thật yên tĩnh."
        binary, setup = selected_binary(model, "stock")
        alternate = (model == "vieneu-cpp" or model in SNAPDRAGON_MODELS or
                     model == "kokoro-litert-omni-gpu" or model.startswith("kokoro-soniqo-"))
        if alternate:
            destination = f"pcm/{model}-smoke"
            setup += " && mkdir -p " + shlex.quote(destination)
            arguments = (binary, family, MODELS[model], f"plans/{family}.tsv",
                         "stream", "2", "0", "0", "1", "1", "3000", destination)
        else:
            arguments = (binary, "synthesize", family, MODELS[model], "2", text, model + "-smoke.wav")
        remote = f"cd {DEVICE} && {setup} && " + shlex.join(arguments)
        try:
            result = adb("shell", remote, timeout=80, check=False)
            (OUTPUT / f"{model}-smoke.log").write_bytes(result.stdout + result.stderr)
            print(f"{model} smoke exited {result.returncode}", flush=True)
            if result.returncode == 0 and (not alternate or any(
                    row.startswith(b"{") and json.loads(row).get("complete_ms") is not None
                    for row in result.stdout.splitlines())):
                available.add(model)
        except subprocess.TimeoutExpired as error:
            detail = (str(error) + "\n").encode()
            (OUTPUT / f"{model}-smoke.log").write_bytes(detail + (error.stdout or b"") + (error.stderr or b""))
            print(f"{model} smoke timed out", flush=True)
            adb("shell", f"pkill -f {shlex.quote('^' + re.escape(binary))} || true", check=False)
    return available


def verify_paired_smokes(arms, pairs, available):
    configurations = {arm[0]: arm for arm in arms}
    failed = set()
    retained = {}
    for reference, candidate in pairs:
        hashes = {}
        if configurations[reference][2] != configurations[candidate][2]:
            raise ValueError("paired smoke arms must use the same delivery mode")
        expected_chunks = 2 if configurations[reference][2] == "stream" else 1
        for label in (reference, candidate):
            arm = configurations[label]
            if arm[1] not in available:
                break
            if label in retained:
                hashes[label] = retained[label]
                continue
            smoke_label = label + "-parity"
            smoke_arm = [smoke_label, *arm[1:]]
            try:
                if run_cell(smoke_arm, {"waves": 1, "period_ms": 3000, "mode": arm[2]}, 0, 1):
                    break
                timings = OUTPUT / f"{smoke_label}.jsonl"
                row = json.loads(timings.read_bytes())
                if row["complete_ms"] is None or len(row["chunks"]) != expected_chunks:
                    break
                directory = shlex.quote(f"{DEVICE}/pcm/{smoke_label}")
                result = adb("shell", f"cd {directory} && sha256sum ./*.s16le")
                hashes[label] = result.stdout.decode().splitlines()
                retained[label] = hashes[label]
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                adb("shell", "pkill -x iktts7s3a5q7o || true", check=False)
                break
        passed = len(hashes.get(reference, ())) == expected_chunks and hashes.get(reference) == hashes.get(candidate)
        (OUTPUT / f"{candidate}-parity-smoke.json").write_text(json.dumps({
            "reference": reference, "candidate": candidate,
            "wave_zero_sha256": hashes, "verdict": "PASS" if passed else "FAIL",
        }, indent=2) + "\n")
        print(f"{candidate} parity smoke {'PASS' if passed else 'FAIL'}", flush=True)
        if not passed:
            failed.add(candidate)
    return failed


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, default=ROOT / "run-plan.json")
    parser.add_argument("--payload", type=Path, default=ROOT / "payload")
    parser.add_argument("--plans", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    plan_directory = args.plans or args.payload / "plans"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.scenario.read_text())
    if not LABEL.fullmatch(manifest["phase"]):
        raise ValueError("invalid scenario phase")
    if any(not LABEL.fullmatch(arm[0]) for arm in manifest["arms"] + manifest.get("profiles", [])):
        raise ValueError("invalid scenario arm label")
    by_label = {arm[0]: arm for arm in manifest["arms"]}
    if len(by_label) != len(manifest["arms"]):
        raise ValueError("scenario arm labels must be unique")
    arm_levels = manifest.get("arm_levels", {})
    if any(label not in by_label or not set(levels) <= set(manifest["levels"])
           for label, levels in arm_levels.items()):
        raise ValueError("arm-specific load levels must be registered in the scenario")
    round_orders = manifest.get("round_orders")
    if round_orders is not None and (
            len(round_orders) != manifest["rounds"] or
            any(len(order) != len(by_label) or set(order) != by_label.keys() for order in round_orders)):
        raise ValueError("every registered round order must be a permutation of the scenario arms")
    thermal_start = manifest.get("thermal_start")
    if thermal_start is not None and (
            thermal_start["max_status"] not in range(7) or
            not 0 < thermal_start["interval_s"] <= thermal_start["timeout_s"] <= 180):
        raise ValueError("thermal conditioning requires a valid status, poll interval and bounded timeout")
    fixture_set = manifest.get("fixture_set")
    if fixture_set is not None and not LABEL.fullmatch(fixture_set):
        raise ValueError("invalid fixture set name")
    if args.plans is not None:
        fixture_directory = args.plans / fixture_set if fixture_set else args.plans
        (args.payload / "plans").mkdir(parents=True, exist_ok=True)
        for family in ("kokoro", "vieneu"):
            name = f"{family}.tsv"
            source = fixture_directory / name
            destination = args.payload / "plans" / name
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
    shutil.copyfile(args.scenario, OUTPUT / "run-plan.json")
    adb_metadata("phone.txt",
                 "getprop ro.product.model; getprop ro.product.manufacturer; getprop ro.product.cpu.abi",
                 "getprop ro.soc.model; getprop ro.soc.manufacturer; getprop ro.hardware",
                 "getprop ro.build.version.release; getprop ro.build.version.sdk; getprop ro.build.fingerprint",
                 "uname -a", "cat /proc/cpuinfo", "grep Cpus_allowed_list /proc/self/status",
                 "for f in /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq; do echo $f; cat $f; done")
    if manifest.get("accelerator_inventory"):
        adb_metadata("phone-accelerators.txt", "cmd gpu vkjson",
                     "getprop ro.hardware.vulkan; getprop ro.soc.model",
                     "ls -l /vendor/lib64/libOpenCL.so 2>/dev/null || true",
                     "grep -i opencl /vendor/etc/public.libraries.txt /system/etc/public.libraries.txt 2>/dev/null || true",
                     "ls -1 /vendor/lib64/libQnnHtp*.so 2>/dev/null || true")
    prepare(args.payload, manifest["arms"])
    adb("shell", f"mkdir -p {DEVICE}")
    push = adb("push", str(args.payload) + "/.", DEVICE, timeout=360, check=False)
    (OUTPUT / "push.log").write_bytes(push.stdout + push.stderr)
    push.check_returncode()
    adb("shell", f"chmod 755 {DEVICE}/bin/iktts7s3a5q7o")
    if (args.payload / "vieneu-cpp").is_dir():
        adb("shell", f"chmod 755 {DEVICE}/vieneu-cpp/capture-vieneu-cpp")
    if (args.payload / "vieneu-snapdragon").is_dir():
        adb("shell", f"chmod 755 {DEVICE}/vieneu-snapdragon/capture-vieneu-cpp")
        if any(arm[1] in {"vieneu-cpp-snap-opencl", "vieneu-cpp-snap-mixed"} for arm in manifest["arms"]):
            adb("shell", f"if [ -r /vendor/lib64/libOpenCL.so ]; then "
                         f"ln -sfn /vendor/lib64/libOpenCL.so {DEVICE}/vieneu-snapdragon/lib/libOpenCL.so; fi")
    if (args.payload / "lib/vieneu-shared").is_dir():
        adb("shell", f"chmod 755 {DEVICE}/lib/vieneu-shared/capture")
    if (args.payload / "soniqo-kokoro").is_dir():
        adb("shell", f"chmod 755 {DEVICE}/soniqo-kokoro/capture-soniqo-kokoro")
    if (args.payload / "omni-kokoro").is_dir():
        adb("shell", f"chmod 755 {DEVICE}/omni-kokoro/capture-omni-kokoro && "
                     f"if [ -r /vendor/lib64/libOpenCL.so ]; then "
                     f"ln -sfn /vendor/lib64/libOpenCL.so {DEVICE}/omni-kokoro/lib/libOpenCL.so; fi")
    if (args.payload / "kokoro-adreno").is_dir():
        adb("shell", f"chmod 755 {DEVICE}/kokoro-adreno/Kokoro_82M_inference_fp16 && "
                     f"ln -sfn {DEVICE}/models/kokoro-multi-lang-v1_0/espeak-ng-data "
                     f"{DEVICE}/kokoro-adreno/assets/espeak-ng-data && "
                     f"if [ -r /vendor/lib64/libOpenCL.so ]; then "
                     f"ln -sfn /vendor/lib64/libOpenCL.so {DEVICE}/kokoro-adreno/libOpenCL.so; fi")
    try:
        available = smoke_models(manifest["arms"], args.payload)
        if not available:
            raise RuntimeError("no model completed native synthesis")
        smoke_failures = verify_paired_smokes(manifest["arms"], manifest.get("parity_smokes", []), available)
        failed_at = dict.fromkeys(smoke_failures, 1)
        for round_number in range(manifest["rounds"]):
            arms = manifest["arms"]
            if round_orders is not None:
                arms = [by_label[label] for label in round_orders[round_number]]
            else:
                offset = round_number * 2 % len(arms)
                arms = arms[offset:] + arms[:offset]
            for concurrency in manifest["levels"]:
                for arm in arms:
                    label, model, mode, _, _, *options = arm
                    if (model not in available or concurrency not in arm_levels.get(label, manifest["levels"]) or
                            concurrency >= failed_at.get(label, float("inf"))):
                        continue
                    family = model.split("-")[0]
                    default_plan = f'{manifest["phase"]}-{family}-{mode}.json'
                    name = options[0].get("INFERENCEKIT_PLAN", default_plan) if options else default_plan
                    if not PLAN_FILE.fullmatch(name):
                        raise ValueError("invalid plan filename")
                    plan = json.loads((plan_directory / name).read_text())
                    if concurrency not in plan["levels"] or round_number >= plan["rounds"]:
                        continue
                    try:
                        if thermal_start is not None:
                            condition_thermal(f"{label}-r{round_number}-n{concurrency}", thermal_start)
                        if run_cell(arm, plan, round_number, concurrency):
                            failed_at[label] = concurrency
                    except subprocess.TimeoutExpired as error:
                        failed_at[label] = concurrency
                        print(f"{label}-r{round_number}-n{concurrency} timed out", flush=True)
                        (OUTPUT / f"{label}-r{round_number}-n{concurrency}-timeout.txt").write_text(str(error))
                        # Match the unique executable staged for this run.
                        adb("shell", "pkill -x iktts7s3a5q7o || true", check=False)
                        if options and options[0].get("INFERENCEKIT_AUDIORUNTIME") == "vieneu-shared":
                            adb("shell", "pkill -f '^./lib/vieneu-shared/capture' || true", check=False)
                        if model == "vieneu-cpp":
                            adb("shell", "pkill -f '^./vieneu-cpp/capture-vieneu-cpp' || true", check=False)
                        elif model in SNAPDRAGON_MODELS:
                            adb("shell", "pkill -f '^./vieneu-snapdragon/capture-vieneu-cpp' || true", check=False)
                        elif model in MEDIAPIPE_MODELS:
                            adb("shell", f"am force-stop {media_app(model)}", check=False)
                        elif model.startswith("kokoro-soniqo-"):
                            adb("shell", "pkill -f '^./soniqo-kokoro/capture-soniqo-kokoro' || true",
                                check=False)
                        elif model == "kokoro-litert-omni-gpu":
                            adb("shell", "pkill -f '^./omni-kokoro/capture-omni-kokoro' || true", check=False)
        for arm in manifest.get("profiles", []):
            label, model, mode, *_ = arm
            if model in available:
                family = model.split("-")[0]
                plan = json.loads((plan_directory / f'{manifest["phase"]}-{family}-{mode}.json').read_text())
                brief = dict(plan, waves=min(plan["waves"], 2))
                run_cell(arm, brief, 0, 1)
        if manifest.get("local_reproduction"):
            verify_local_package(args.payload.resolve())
    finally:
        adb_metadata("phone-after.txt", "dumpsys thermalservice")
        pull = adb("pull", f"{DEVICE}/pcm", str(OUTPUT / "pcm"), timeout=180, check=False)
        (OUTPUT / "pull.log").write_bytes(pull.stdout + pull.stderr)
        with tarfile.open(OUTPUT.parent / "speech-capture.tar.gz", "w:gz", compresslevel=1) as archive:
            archive.add(OUTPUT, arcname="speech-capture")


def test_capture():
    main()


if __name__ == "__main__":
    main()
