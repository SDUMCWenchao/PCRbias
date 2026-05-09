#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, json, math
from collections import defaultdict
from pathlib import Path

try:
    import orjson  # type: ignore
    def loads(s: str): return orjson.loads(s)
except Exception:
    def loads(s: str): return json.loads(s)

def read_kmer_posbins(fp: Path):
    m = {}
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            sid = row["Seq_ID"]
            js = row["kmer_json"] or "{}"
            try:
                m[sid] = loads(js)
            except Exception:
                m[sid] = {}
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--chunk_name", required=True)
    ap.add_argument("--x_mode", choices=["count","presence"], default="count")
    args = ap.parse_args()

    project = Path(args.project_dir)
    train_fp = project / "analysis_results" / "03_DataWeaver" / "training_chunks" / f"{args.chunk_name}.train.tsv.gz"
    kmer_fp_new = project / "analysis_results" / "02_Features" / "kmer_region_chunks" / f"{args.chunk_name}.kmer_regions.tsv.gz"
    kmer_fp_old = project / "analysis_results" / "02_Features" / "kmer_posbins_chunks" / f"{args.chunk_name}.kmer_posbins.tsv.gz"

    kmer_fp = kmer_fp_new if kmer_fp_new.exists() else kmer_fp_old
    if not kmer_fp.exists():
       raise FileNotFoundError(f"Missing kmer features: {kmer_fp_new} (or old fallback {kmer_fp_old})")


    if not train_fp.exists():
        raise FileNotFoundError(train_fp)
    if not kmer_fp.exists():
        raise FileNotFoundError(kmer_fp)

    out_dir = project / "analysis_results" / "04_Stats_Kmer"
    base_dir = out_dir / "bases"
    part_dir = out_dir / "partials"
    base_dir.mkdir(parents=True, exist_ok=True)
    part_dir.mkdir(parents=True, exist_ok=True)

    out_base = base_dir / f"{args.chunk_name}.base.tsv.gz"
    out_part = part_dir / f"{args.chunk_name}.kmer.partial.tsv.gz"
    if out_base.exists() and out_base.stat().st_size > 0 and out_part.exists() and out_part.stat().st_size > 0:
        print(f"[SKIP] {args.chunk_name} partials exist")
        return

    seq2k = read_kmer_posbins(kmer_fp)

    # base[pair] = [W, ysum, y2sum, cy_total, cn_total, nrows]
    base = defaultdict(lambda: [0.0,0.0,0.0, 0.0,0.0, 0])

    # feat[(pair,feat)] = [xw,x2w,xyw, xcy,x2cy, xcn,x2cn]
    feat = defaultdict(lambda: [0.0,0.0,0.0, 0.0,0.0, 0.0,0.0])

    with gzip.open(train_fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pid = row["pair_id"]
            sid = row["Seq_ID"]
            y = float(row["log2fc"])
            cy = int(row["count_yes"])
            cn = int(row["count_no"])
            w = float(cy + cn)
            if w <= 0:
                continue

            b = base[pid]
            b[0] += w
            b[1] += w * y
            b[2] += w * y * y
            b[3] += cy
            b[4] += cn
            b[5] += 1

            km = seq2k.get(sid, {})
            if not km:
                continue

            for kfeat, xv in km.items():
                x = float(xv)
                if args.x_mode == "presence":
                    x = 1.0
                if x <= 0:
                    continue
                a = feat[(pid, kfeat)]
                a[0] += w * x
                a[1] += w * x * x
                a[2] += w * x * y
                if cy > 0:
                    a[3] += cy * x
                    a[4] += cy * x * x
                if cn > 0:
                    a[5] += cn * x
                    a[6] += cn * x * x

    with gzip.open(out_base, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","W","y_sum","y2_sum","cy_total","cn_total","n_rows"])
        for pid, a in base.items():
            w.writerow([pid, f"{a[0]:.12g}", f"{a[1]:.12g}", f"{a[2]:.12g}", f"{a[3]:.12g}", f"{a[4]:.12g}", a[5]])

    with gzip.open(out_part, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","feature","xw","x2w","xyw","xcy","x2cy","xcn","x2cn"])
        for (pid, fkey), a in feat.items():
            w.writerow([pid, fkey,
                        f"{a[0]:.12g}", f"{a[1]:.12g}", f"{a[2]:.12g}",
                        f"{a[3]:.12g}", f"{a[4]:.12g}",
                        f"{a[5]:.12g}", f"{a[6]:.12g}"])

    print(f"[DONE] {args.chunk_name}: base_pairs={len(base)} feat_rows={len(feat)}")

if __name__ == "__main__":
    main()
