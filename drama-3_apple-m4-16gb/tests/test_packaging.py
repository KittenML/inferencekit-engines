"""CPU checks for the retained build, serving and smoke dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ("main5-tower5", "main6-tower5")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PackagingTests(unittest.TestCase):
    def test_parent_installer_matches_pinned_source(self):
        parent = ROOT / "profiles/main6-tower5/source/parent-installer-source"
        for path in parent.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (ROOT / path.relative_to(parent)).read_bytes())

    def test_runtime_assets_and_selected_implementation(self):
        for profile in PROFILES:
            runtime = ROOT / "profiles" / profile / "source/measured"
            recipe = json.loads((runtime / "selected_recipe.json").read_text())
            for name, expected in recipe["implementation"]["files"].items():
                self.assertEqual(digest(runtime / name), expected)
            settings = recipe["runtime"]
            for key in ("fixed_reference_manifest", "startup_warmup_fixture", "startup_warmup_native_proof"):
                hash_key = "fixed_reference_sha256" if key == "fixed_reference_manifest" else key + "_sha256"
                self.assertEqual(digest(runtime / settings[key]), settings[hash_key])
            cache = runtime / "conditioning-cache-v2"
            reference = json.loads((cache / "manifest.json").read_text())
            for file_key, hash_key in (("wav_file", "wav_sha256"), ("codes_file", "codes_file_sha256"),
                                      ("asset_admission_evidence_file", "asset_admission_evidence_sha256"),
                                      ("native_initial_prompt_proof_file", "native_initial_prompt_proof_sha256")):
                self.assertEqual(digest(cache / reference[file_key]), reference[hash_key])
            self.assertEqual(digest(runtime / "registered_client.py"),
                             "450a95b355d7343136dbd67c1047713d8c3d92eef8a4e6eccd2b122641a9dd0c")
            fixtures = [json.loads(line) for line in (runtime / "fixtures/encoder-release-public8-v14-bench.jsonl").read_text().splitlines()]
            self.assertEqual(next(row for row in fixtures if row["id"] == "screen_short")["text"],
                             "The copper latch clicked twice before the door opened.")

    def test_staging_and_tamper_rejection(self):
        # Separate interpreters avoid sharing same-named profile modules. No model is loaded.
        script = '''
import json, shutil, sys, tempfile
from pathlib import Path
from unittest.mock import patch
source = Path(sys.argv[1])
sys.path.insert(0, str(source))
import portable_common as common
import profile_setup as setup
common.verify_packet()
with tempfile.TemporaryDirectory() as folder:
    work = Path(folder)
    parent, original = work / "parent", work / "original"
    parent.mkdir(); original.mkdir()
    if source.parent.name == "main5-tower5":
        shutil.copyfile(source / "PROTECTED_TOWER5_ORIGIN.json", parent / "origin.json")
    with patch.object(setup, "disk"):
        setup.stage(str(work / "staged"), sys.executable, str(parent), str(original), 120)
    runtime = work / "staged/runtime"
    common.verify_measured(runtime)
    (runtime / "sampler_sort.py").write_text("tampered")
    try:
        common.verify_measured(runtime)
    except ValueError:
        pass
    else:
        raise AssertionError("Changed inference source was accepted")
    # Each profile has a distinct module identity; verify the wrapper's pinned seal.
    import runpy
    wrapper = runpy.run_path(str(source.parent / "serve_guarded.py"))
    assert wrapper["SOURCE"] == common.sha(source / "SOURCE_SEAL.json")
'''
        for profile in PROFILES:
            source = ROOT / "profiles" / profile / "source"
            result = subprocess.run([sys.executable, "-B", "-c", script, str(source)],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_personal_workspace_paths(self):
        for path in ROOT.rglob("*.json"):
            self.assertNotIn("/home/haaziq/", path.read_text(), str(path))

    def test_python_sources_parse(self):
        for path in ROOT.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path))


if __name__ == "__main__":
    unittest.main()
