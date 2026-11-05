#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
python_bin="$(command -v python3.12)"
command -v git >/dev/null
command -v patch >/dev/null
command -v dpkg-deb >/dev/null
env -i HOME="${HOME:?HOME required}" PATH=/usr/bin:/bin LANG=C.UTF-8 "$python_bin" -E "$root/verify_snapshot.py"
env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 PIP_CONFIG_FILE=/dev/null "$python_bin" -E "$root/install_runtime.py"
exec env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 "$python_bin" -E "$root/fp8_install.py"
