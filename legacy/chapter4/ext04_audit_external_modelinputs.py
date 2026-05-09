#!/usr/bin/env python3
import json
from pathlib import Path

root = Path("analysis_results/05_ModelInputs_external_topbias")
bad = []
total = 0

for info_fp in root.rglob("*/kmer_only/dataset_info.json"):
    total += 1
    info = json.loads(info_fp.read_text())
    nnz = info.get("kmer_nnz", {})
    s = nnz.get("train", 0) + nnz.get("val", 0) + nnz.get("test", 0)
    if s == 0:
        bad.append(str(info_fp))

print("[INFO] datasets =", total)
print("[INFO] bad(kmer nnz=0) =", len(bad))
for x in bad[:30]:
    print("[BAD]", x)
if len(bad) > 30:
    print("[INFO] ... truncated ...")
