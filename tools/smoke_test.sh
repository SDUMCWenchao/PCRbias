#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python - <<'PY'
from pathlib import Path
import py_compile
import sys

errors = []
for path in sorted(list(Path('legacy').rglob('*.py')) + list(Path('tools').glob('*.py'))):
    try:
        py_compile.compile(str(path), doraise=True)
    except Exception as exc:
        errors.append((path, exc))

if errors:
    for path, exc in errors:
        print(f"PY_ERROR {path}: {exc}")
    sys.exit(1)
print("Python syntax check passed.")
PY

SH_ERR=0
while IFS= read -r -d '' f; do
  bash -n "$f" || SH_ERR=1
done < <(find legacy tools \( -name '*.sh' -o -name '*.slurm' \) -print0)

if [[ "$SH_ERR" -ne 0 ]]; then
  echo "Shell syntax check failed."
  exit 1
fi

echo "Smoke test passed: Python and shell syntax are valid."
