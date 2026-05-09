#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, glob
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext_root", required=True)
    args = ap.parse_args()

    ext = Path(args.ext_root).resolve()
    part_dir = ext / "analysis_results/02_Features/partials"
    out = ext / "analysis_results/02_Features/features_seq.tsv.gz"

    files = sorted(part_dir.glob("feat_chunk_*.tsv.gz"))
    if not files:
        raise FileNotFoundError(f"no partials in {part_dir}")

    dfs = []
    for p in files:
        df = pd.read_csv(p, sep="\t", compression="gzip")
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # 约定必须有 Seq_ID
    if "Seq_ID" not in all_df.columns:
        raise ValueError("features missing Seq_ID")
    all_df = all_df.drop_duplicates("Seq_ID")

    all_df.to_csv(out, sep="\t", index=False, compression="gzip")
    print("[DONE]", out, "rows=", len(all_df), "cols=", all_df.shape[1])

if __name__ == "__main__":
    main()
