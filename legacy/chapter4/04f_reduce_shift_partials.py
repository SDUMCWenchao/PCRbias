#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

def read_partial(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row

def w_mean_var(w, s1, s2):
    if w <= 0:
        return (float("nan"), float("nan"))
    m = s1 / w
    v = s2 / w - m*m
    if v < 0:
        v = 0.0
    return m, v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--out_name", default="pair_feature_shift_top200.tsv.gz")
    args = ap.parse_args()

    project = Path(args.project_dir)
    part_dir = project / "analysis_results" / "04_Stats" / "shift_partials"
    out_dir  = project / "analysis_results" / "04_Stats"
    pairs_fp = project / "analysis_results" / "03_DataWeaver" / "pairs.tsv"

    files = sorted(part_dir.glob("chunk_*.shift.partial.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"No shift partials in {part_dir}")

    # pair meta map
    pair_meta: Dict[str, Tuple[str,str,str,str,str]] = {}
    with pairs_fp.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            pair_meta[row["pair_id"]] = (row["species"], row["n_individuals"], row["locus"], row["yes_file_id"], row["no_file_id"])

    # sums[(pair,feat)] = [wy,sumy,sumy2, wn,sumn,sumn2, ny, nn]
    sums: Dict[Tuple[str,str], list] = defaultdict(lambda: [0.0,0.0,0.0, 0.0,0.0,0.0, 0,0])

    for fp in files:
        for row in read_partial(fp):
            pid = row["pair_id"]; feat = row["feature"]
            a = sums[(pid, feat)]
            a[0] += float(row["wy"]);   a[1] += float(row["sumy"]);  a[2] += float(row["sumy2"])
            a[3] += float(row["wn"]);   a[4] += float(row["sumn"]);  a[5] += float(row["sumn2"])
            a[6] += int(row["ny"]);     a[7] += int(row["nn"])

    out_fp = out_dir / args.out_name
    with gzip.open(out_fp, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow([
            "pair_id","species","n_individuals","locus","yes_file_id","no_file_id",
            "feature","wy","wn","mean_yes","mean_no","delta_yes_minus_no",
            "sd_yes","sd_no","effect_d",
            "ny_rows","nn_rows"
        ])

        for (pid, feat), a in sums.items():
            species, nind, locus, yfid, nfid = pair_meta.get(pid, ("NA","NA","NA","NA","NA"))
            wy,sumy,sumy2, wn,sumn,sumn2, ny, nn = a

            my, vy = w_mean_var(wy, sumy, sumy2)
            mn, vn = w_mean_var(wn, sumn, sumn2)
            dy = math.sqrt(vy) if vy==vy else float("nan")
            dn = math.sqrt(vn) if vn==vn else float("nan")

            if my==my and mn==mn:
                delta = my - mn
            else:
                delta = float("nan")

            pooled = None
            if vy==vy and vn==vn:
                pooled = math.sqrt((vy + vn)/2.0)
            d = (delta / pooled) if (pooled and pooled > 0 and delta==delta) else float("nan")

            w.writerow([
                pid, species, nind, locus, yfid, nfid,
                feat,
                f"{wy:.12g}", f"{wn:.12g}",
                f"{my:.12g}" if my==my else "NA",
                f"{mn:.12g}" if mn==mn else "NA",
                f"{delta:.12g}" if delta==delta else "NA",
                f"{dy:.12g}" if dy==dy else "NA",
                f"{dn:.12g}" if dn==dn else "NA",
                f"{d:.12g}" if d==d else "NA",
                ny, nn
            ])

    print(f"[DONE] {out_fp} rows={len(sums)}")

if __name__ == "__main__":
    main()
