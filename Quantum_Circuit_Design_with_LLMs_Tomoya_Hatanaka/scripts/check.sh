#!/usr/bin/env bash
# Local CI: lint + test. Run before every commit.
# Works whether or not a virtualenv is already activated.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -n "${VIRTUAL_ENV:-}" ]; then
    PYTHON="python"
elif [ -x ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "==> ruff check"
"$PYTHON" -m ruff check .

echo "==> pytest"
"$PYTHON" -m pytest tests/ -q

echo "==> all checks passed"
