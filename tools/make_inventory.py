#!/usr/bin/env python3
"""Regenerate docs/SCRIPT_INVENTORY.tsv."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "SCRIPT_INVENTORY.tsv"
rows = []
for p in sorted(ROOT.rglob("*")):
    if not p.is_file() or ".git" in p.parts:
        continue
    try:
        lines = len(p.read_text(errors="ignore").splitlines())
    except Exception:
        lines = 0
    if p.suffix == ".py":
        typ = "python"
    elif p.suffix == ".sh":
        typ = "shell"
    elif p.suffix == ".slurm":
        typ = "slurm"
    elif p.suffix == ".R":
        typ = "r"
    elif p.suffix == ".tsv":
        typ = "tsv"
    else:
        typ = p.suffix.lstrip(".") or "file"
    rows.append((p.relative_to(ROOT), typ, lines))

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    f.write("path\ttype\tlines\n")
    for row in rows:
        f.write("\t".join(map(str, row)) + "\n")
print(f"Wrote {OUT}")
