#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

def load_features(path: Path) -> List[str]:
    feats = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                feats.append(s)
    return feats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--chunk_name", required=True)  # chunk_0001
    ap.add_argument("--feature_list", required=True)  # top_features_200.txt
    ap.add_argument("--min_count", type=int, default=1, help="ignore weights < min_count")
    args = ap.parse_args()

    project = Path(args.project_dir)
    in_fp = project / "analysis_results" / "03_DataWeaver" / "training_chunks" / f"{args.chunk_name}.train.tsv.gz"
    if not in_fp.exists():
        raise FileNotFoundError(in_fp)

    out_dir = project / "analysis_results" / "04_Stats" / "shift_partials"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{args.chunk_name}.shift.partial.tsv.gz"
    if out_fp.exists() and out_fp.stat().st_size > 0:
        print(f"[SKIP] {out_fp.name} exists")
        return

    feats = load_features(Path(args.feature_list))
    feats_set = set(feats)

    # accum[(pair, feat)] = [wy,sumy,sumy2, wn,sumn,sumn2, ny, nn]
    accum: Dict[Tuple[str,str], List[float]] = defaultdict(lambda: [0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0])

    with gzip.open(in_fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pid = row["pair_id"]
            cy = int(row["count_yes"])
            cn = int(row["count_no"])
            if cy < args.min_count and cn < args.min_count:
                continue

            for feat in feats:
                v = row.get(feat, "")
                if v == "" or v is None:
                    continue
                try:
                    x = float(v)
                except Exception:
                    continue

                a = accum[(pid, feat)]
                if cy >= args.min_count:
                    wy = float(cy)
                    a[0] += wy
                    a[1] += wy * x
                    a[2] += wy * x * x
                    a[6] += 1.0
                if cn >= args.min_count:
                    wn = float(cn)
                    a[3] += wn
                    a[4] += wn * x
                    a[5] += wn * x * x
                    a[7] += 1.0

    with gzip.open(out_fp, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","feature","wy","sumy","sumy2","wn","sumn","sumn2","ny","nn"])
        for (pid, feat), a in accum.items():
            w.writerow([pid, feat,
                        f"{a[0]:.12g}", f"{a[1]:.12g}", f"{a[2]:.12g}",
                        f"{a[3]:.12g}", f"{a[4]:.12g}", f"{a[5]:.12g}",
                        int(a[6]), int(a[7])])

    print(f"[DONE] {args.chunk_name}: rows={len(accum)} -> {out_fp.name}")

if __name__ == "__main__":
    main()
