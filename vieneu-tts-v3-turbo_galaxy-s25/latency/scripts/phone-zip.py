"""Add the DEX, asset tree, and extracted Android arm64 binaries to the signed APK input."""

from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


def main() -> None:
    apk, dex, runner, payload, plans = map(Path, sys.argv[1:])
    assets = sorted(path for path in payload.rglob("*") if path.is_file())
    if not assets:
        raise SystemExit("No runtime payload files")
    with ZipFile(apk, "a") as package:
        package.write(dex, "classes.dex", ZIP_DEFLATED)
        for lib in sorted(runner.glob("lib*.so")):
            with lib.open("rb") as stream:
                header = stream.read(20)
            if header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\xb7\0":
                raise SystemExit(f"Expected an Android ARM64 ELF binary: {lib.name}")
            package.write(lib, f"lib/arm64-v8a/{lib.name}", ZIP_STORED)
        for path in sorted(plans.glob("*.tsv")):
            package.write(path, f"assets/{path.name}", ZIP_DEFLATED)
        for path in assets:
            package.write(path, f"assets/payload/{path.relative_to(payload).as_posix()}", ZIP_STORED)


if __name__ == "__main__":
    main()
