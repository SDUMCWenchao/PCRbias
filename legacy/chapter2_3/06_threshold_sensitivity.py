#!/usr/bin/env python3
from pathlib import Path
import argparse
import math
import pandas as pd
import numpy as np

def shannon(p):
    p = np.asarray([x for x in p if x > 0], dtype=float)
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0

def simpson_1_minus_d(p):
    p = np.asarray([x for x in p if x > 0], dtype=float)
    return float(1.0 - np.square(p).sum()) if len(p) else 0.0

def summarize(df, threshold_label):
    rows = []
    for keys, sub in df.groupby(["sample_id","marker","group_name","sample_type","species_scope","is_core_analysis"], dropna=False):
        total_reads_all = sub["count"].sum()
        kept = sub[sub["keep"]].copy()
        total_reads_kept = kept["count"].sum()
        p = kept["count"] / total_reads_kept if total_reads_kept > 0 else pd.Series(dtype=float)
        richness = kept["sequence_id"].nunique()
        dominant = float(kept["relative_abundance"].max()) if len(kept) else 0.0
        top10 = float(kept["relative_abundance"].sort_values(ascending=False).head(10).sum()) if len(kept) else 0.0
        rows.append({
            "sample_id": keys[0],
            "marker": keys[1],
            "group_name": keys[2],
            "sample_type": keys[3],
            "species_scope": keys[4],
            "is_core_analysis": keys[5],
            "threshold_label": threshold_label,
            "n_unique_sequences_kept": richness,
            "total_reads_all": int(total_reads_all),
            "total_reads_kept": int(total_reads_kept),
            "fraction_reads_kept": 0.0 if total_reads_all == 0 else float(total_reads_kept / total_reads_all),
            "shannon": shannon(p),
            "simpson_1_minus_d": simpson_1_minus_d(p),
            "dominant_rel_abundance": dominant,
            "top10_rel_abundance": top10,
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Threshold sensitivity analysis for master_long_abundance.tsv")
    ap.add_argument("--abundance", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--write-filtered-tables", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.abundance, sep="\t")
    thresholds = {
        "count_ge_2": ("count", 2),
        "rel_ge_0.001": ("relative_abundance", 0.001),
        "rel_ge_0.005": ("relative_abundance", 0.005),
        "rel_ge_0.01": ("relative_abundance", 0.01),
    }

    all_summaries = []
    for label, (col, cutoff) in thresholds.items():
        tmp = df.copy()
        tmp["keep"] = tmp[col] >= cutoff
        all_summaries.append(summarize(tmp, label))
        if args.write_filtered_tables:
            tmp[tmp["keep"]].drop(columns=["keep"]).to_csv(outdir / f"{label}.filtered.tsv", sep="\t", index=False)

    summary_df = pd.concat(all_summaries, ignore_index=True)
    summary_df.to_csv(outdir / "threshold_sensitivity_sample_summary.tsv", sep="\t", index=False)

    wide = summary_df.pivot_table(
        index=["sample_id","marker","group_name","sample_type","species_scope","is_core_analysis"],
        columns="threshold_label",
        values=["n_unique_sequences_kept","fraction_reads_kept","shannon","simpson_1_minus_d","dominant_rel_abundance"],
    )
    wide.columns = ["__".join(map(str, c)) for c in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(outdir / "threshold_sensitivity_wide.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'threshold_sensitivity_sample_summary.tsv'}")
    print(f"Wrote {outdir / 'threshold_sensitivity_wide.tsv'}")

if __name__ == "__main__":
    main()
