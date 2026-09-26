#!/bin/sh
set -eu
studio_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if command -v python3 >/dev/null 2>&1; then
  exec python3 "$studio_dir/launch.py" "$@"
elif command -v python >/dev/null 2>&1; then
  exec python "$studio_dir/launch.py" "$@"
else
  echo "Python 3.11 or newer is required: https://www.python.org/downloads/" >&2
  exit 1
fi
