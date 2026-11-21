"""Schedule one selected Galaxy and download its test artifacts without logging AWS identifiers."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import boto3
from botocore.exceptions import ClientError
import requests

ROOT = Path(__file__).resolve().parent.parent
PRIVATE = ROOT / "private"
CONFIGURATION = PRIVATE / "aws.json"
REGION = "us-west-2"
FARM = boto3.client("devicefarm", region_name=REGION)


def configure(project_name):
    projects = [project for page in FARM.get_paginator("list_projects").paginate()
                for project in page["projects"] if project["name"] == project_name]
    if len(projects) > 1:
        raise RuntimeError("the selected project name is ambiguous")
    project = projects[0] if projects else FARM.create_project(
        name=project_name, defaultJobTimeoutMinutes=22,
    )["project"]
    devices = (device for page in FARM.get_paginator("list_devices").paginate(filters=[
        {"attribute": "MODEL", "operator": "EQUALS", "values": ["Samsung Galaxy S25"]},
    ]) for device in page["devices"])
    choices = [device for device in devices if device["name"] == "Samsung Galaxy S25" and device["os"] == "16"]
    if len(choices) != 1:
        raise RuntimeError("could not uniquely select Samsung Galaxy S25 Android 16")
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    CONFIGURATION.write_text(json.dumps({"project": project["arn"], "device": choices[0]["arn"]}))
    os.chmod(CONFIGURATION, 0o600)
    print(f"Project configured for {choices[0]['name']} Android {choices[0]['os']}", flush=True)


def upload(path, kind):
    content_type = "application/octet-stream"
    details = json.loads(CONFIGURATION.read_text())
    item = FARM.create_upload(projectArn=details["project"], name=path.name,
                              type=kind, contentType=content_type)["upload"]
    with path.open("rb") as source:
        response = requests.put(item["url"], data=source, headers={"Content-Type": content_type},
                                timeout=(30, 600))
    if not response.ok:
        raise RuntimeError(f"{path.name}: upload HTTP {response.status_code}")
    for _ in range(90):
        result = FARM.get_upload(arn=item["arn"])["upload"]
        if result["status"] == "SUCCEEDED":
            print(f"{path.name}: accepted", flush=True)
            return item["arn"]
        if result["status"] == "FAILED":
            metadata = json.loads(result.get("metadata", "{}"))
            raise RuntimeError(f"{path.name}: {metadata.get('errorCode', 'validation failed')}")
        time.sleep(4)
    raise TimeoutError(f"{path.name}: upload validation timed out")


def stage(phase):
    paths = (
        (ROOT / "build/capture.apk", "ANDROID_APP"),
        (ROOT / f"build/{phase}.zip", "APPIUM_PYTHON_TEST_PACKAGE"),
        (ROOT / "source/devicefarm.yml", "APPIUM_PYTHON_TEST_SPEC"),
    )
    digests = []
    for path, _ in paths:
        with path.open("rb") as stream:
            digests.append(hashlib.file_digest(stream, "sha256").hexdigest())
    state_path = PRIVATE / f"{phase}-uploads.json"
    if state_path.exists():
        registered = json.loads(state_path.read_text())
        if registered["digests"] == digests:
            return registered["arns"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        uploads = [pool.submit(upload, path, kind) for path, kind in paths]
        arns = [result.result() for result in uploads]
    state_path.write_text(json.dumps({"digests": digests, "arns": arns}))
    os.chmod(state_path, 0o600)
    return arns


def launch(phase, timeout):
    state_path = PRIVATE / f"{phase}-run.json"
    if state_path.exists():
        print("Run already scheduled", flush=True)
        return
    arns = stage(phase)
    details = json.loads(CONFIGURATION.read_text())
    run = FARM.schedule_run(
        projectArn=details["project"], appArn=arns[0],
        deviceSelectionConfiguration={
            "filters": [{"attribute": "ARN", "operator": "IN", "values": [details["device"]]}],
            "maxDevices": 1,
        },
        name=f"fresh-speech-{phase}",
        test={"type": "APPIUM_PYTHON", "testPackageArn": arns[1], "testSpecArn": arns[2]},
        executionConfiguration={"jobTimeoutMinutes": timeout, "videoCapture": False,
                                "skipAppResign": True},
    )["run"]
    state_path.write_text(json.dumps({"arn": run["arn"]}))
    os.chmod(state_path, 0o600)
    print(run["status"], flush=True)


def get_run(phase):
    arn = json.loads((PRIVATE / f"{phase}-run.json").read_text())["arn"]
    run = FARM.get_run(arn=arn)["run"]
    print(run["status"], run["result"], run.get("deviceMinutes", {}), flush=True)
    return run


def objects(operation, **kwargs):
    paginator = FARM.get_paginator(operation)
    for page in paginator.paginate(**kwargs):
        yield from page.get("artifacts", page.get("jobs", page.get("suites", page.get("tests", []))))


def collect(phase):
    run = get_run(phase)
    output = ROOT / "results" / phase
    output.mkdir(parents=True, exist_ok=True)
    metadata = {key: run[key] for key in
                ("name", "status", "result", "deviceMinutes", "counters", "started", "stopped") if key in run}
    (output / "run-metadata.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
    candidates = [run["arn"]]
    for job in objects("list_jobs", arn=run["arn"]):
        candidates.append(job["arn"])
        for suite in objects("list_suites", arn=job["arn"]):
            candidates.append(suite["arn"])
            for test in objects("list_tests", arn=suite["arn"]):
                candidates.append(test["arn"])
    retrieved = set()
    for arn in candidates:
        for category in ("FILE", "LOG"):
            for artifact in objects("list_artifacts", arn=arn, type=category):
                key = artifact["arn"]
                if key in retrieved:
                    continue
                retrieved.add(key)
                response = requests.get(artifact["url"], timeout=(10, 60), stream=True)
                if not response.ok:
                    print(f"artifact {artifact['name']}: HTTP {response.status_code}", flush=True)
                    continue
                filename = artifact["name"].replace("/", "-")
                extension = artifact.get("extension", "")
                if extension and not filename.endswith("." + extension):
                    filename += "." + extension
                path = output / filename
                if path.exists():
                    suffix = hashlib.sha256(key.encode()).hexdigest()[:10]
                    path = output / f"{suffix}-{filename}"
                destination = tempfile.NamedTemporaryFile(dir=output, prefix=".artifact-", delete=False)
                temporary = Path(destination.name)
                try:
                    with destination:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            destination.write(chunk)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
                print("collected", filename, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("configure", "stage", "launch", "status", "collect"))
    parser.add_argument("phase", nargs="?", default="screen2")
    parser.add_argument("--project-name", default="inferencekit-android-speech")
    parser.add_argument("--timeout", type=int, default=15)
    arguments = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", arguments.phase):
        parser.error("invalid phase name")
    try:
        if arguments.command == "configure":
            configure(arguments.project_name)
        elif arguments.command == "stage":
            stage(arguments.phase)
        elif arguments.command == "launch":
            launch(arguments.phase, arguments.timeout)
        elif arguments.command == "status":
            get_run(arguments.phase)
        else:
            collect(arguments.phase)
    except ClientError as error:
        print(f"{error.operation_name}: {error.response['Error']['Code']}", flush=True)
        raise SystemExit(1) from None
    except requests.exceptions.RequestException as error:
        print(f"artifact transfer failed: {type(error).__name__}", flush=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
