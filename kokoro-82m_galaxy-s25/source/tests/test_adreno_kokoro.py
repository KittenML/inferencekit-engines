import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class AdrenoKokoroFramingTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("c++"), "requires a host C++ compiler")
    def test_logger_prefix_cannot_hide_a_pcm_header(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "Kokoro_82M_inference_fp16"
            runtime.write_text(r'''#!/usr/bin/env python3
import struct
import sys

print(r"driver diagnostic\nready.", file=sys.stderr, flush=True)
assert sys.stdin.readline() == "Hello there.\n"
print(r"unsupported optional kernel\nKOKORO_PCM_BEGIN 42600 24000", file=sys.stderr, flush=True)
sys.stdout.buffer.write(struct.pack("<h", 140) * 42600)
sys.stdout.buffer.flush()
print(r"completed\nKOKORO_UTT_END", file=sys.stderr, flush=True)
''')
            runtime.chmod(0o755)
            client = root / "client.cpp"
            client.write_text(r'''
#include <algorithm>
#include "kokoro_pcm_process.h"

int main(int argc, char **argv) {
    if (argc != 2) {
        return 1;
    }
    KokoroPcmProcess runner(argv[1], {"./Kokoro_82M_inference_fp16"});
    auto pcm = runner.synthesize("Hello there.");
    return pcm.size() == 42600 &&
           std::all_of(pcm.begin(), pcm.end(), [](int16_t sample) { return sample == 140; }) ? 0 : 1;
}
''')
            executable = root / "client"
            subprocess.run(("c++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I", source,
                            client, "-o", executable), check=True, capture_output=True, timeout=30)
            subprocess.run((executable, root), check=True, capture_output=True, timeout=5)
