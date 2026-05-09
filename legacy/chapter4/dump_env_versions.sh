#!/usr/bin/env bash
set -euo pipefail

out="${1:-ENV_versions_$(date +%F).txt}"

{
  echo "## DATE"
  date
  echo

  echo "## SYSTEM"
  uname -a
  echo

  echo "## PYTHON"
  which python || true
  python --version || true
  echo

  echo "## PIP FREEZE"
  python -m pip freeze 2>/dev/null || true
  echo

  echo "## CONDA (if available)"
  conda --version 2>/dev/null || true
  conda info 2>/dev/null || true
  conda list 2>/dev/null || true
  echo

  echo "## KEY TOOLS (optional)"
  for cmd in fastp cutadapt vsearch usearch qiime dada2; do
    command -v "$cmd" >/dev/null 2>&1 && { echo "# $cmd"; "$cmd" --version 2>&1 | head; echo; }
  done
} > "$out"

echo "[OK] Wrote $out"
