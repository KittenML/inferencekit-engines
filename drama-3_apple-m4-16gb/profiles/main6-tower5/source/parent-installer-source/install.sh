#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu
cd "$(dirname "$0")"
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo 'Native arm64 macOS required' >&2; exit 1; }
mkdir -p .tools .cache
UV_ARCHIVE=.tools/uv-0.12.7-aarch64-apple-darwin.tar.gz
if [ ! -f "$UV_ARCHIVE" ]; then
  /usr/bin/curl --fail --location --retry 3 --output "$UV_ARCHIVE" https://github.com/astral-sh/uv/releases/download/0.12.7/uv-aarch64-apple-darwin.tar.gz
fi
printf '%s\n' "127ebdda7ad953cdf198e964b570ea5771b85467ea93eb7cb6d6f8e6f55408f3  $UV_ARCHIVE" | /usr/bin/shasum -a 256 -c -
/usr/bin/tar -xzf "$UV_ARCHIVE" -C .tools
UV_BIN="$PWD/.tools/uv-aarch64-apple-darwin/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
export UV_CACHE_DIR="$PWD/.cache/uv"
"$UV_BIN" python install 3.12.14
"$UV_BIN" venv --python 3.12.14 .venv
"$UV_BIN" pip sync --python .venv/bin/python requirements.runtime.lock
.venv/bin/python prepare_model.py "$@"
