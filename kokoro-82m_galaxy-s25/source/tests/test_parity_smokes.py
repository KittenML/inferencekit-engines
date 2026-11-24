import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_device


class ParitySmokeTest(unittest.TestCase):
    def test_streaming_plan_reuses_one_reference_across_candidates(self):
        model = "kokoro-mediapipe-gpu"
        names = ("kokoro-reference", "kokoro-first", "kokoro-second")
        arms = [[name, model, "stream", 0, 0] for name in names]
        pairs = [(names[0], name) for name in names[1:]]
        captured = []
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def capture(arm, plan, round_number, concurrency):
                self.assertEqual(plan["mode"], "stream")
                self.assertEqual(round_number, 0)
                self.assertEqual(concurrency, 1)
                captured.append(arm[0])
                with (output / f"{arm[0]}.jsonl").open("a") as rows:
                    rows.write(json.dumps({"complete_ms": 300, "chunks": [[0], [1]]}) + "\n")
                return 0

            def checksums(*args):
                first = b"abcd" if names[2] not in args[1] else b"dcba"
                return subprocess.CompletedProcess(args, 0, first + b"  chunk0\n1234  chunk1\n")

            with patch.object(run_device, "OUTPUT", output), patch.object(
                    run_device, "run_cell", side_effect=capture), patch.object(
                    run_device, "adb", side_effect=checksums):
                failed = run_device.verify_paired_smokes(arms, pairs, {model})

            self.assertEqual(captured, [name + "-parity" for name in names])
            self.assertEqual(failed, {names[2]})


if __name__ == "__main__":
    unittest.main()
