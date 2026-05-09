#!/usr/bin/env python3
from pathlib import Path
import argparse, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--qc-summary", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args()
outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.qc_summary, sep="\t")
df["retention_rate"] = df["retention_rate"].astype(float)
overall = pd.DataFrame({
    "n_samples":[df["sample_id"].nunique()],
    "raw_reads_total":[df["raw_reads"].sum()],
    "filtered_reads_total":[df["filtered_reads"].sum()],
    "mean_retention_rate":[df["retention_rate"].mean()],
    "median_retention_rate":[df["retention_rate"].median()],
})
overall.to_csv(outdir / "qc_overall_summary.tsv", sep="\t", index=False)
by_marker = df.groupby("marker", as_index=False).agg(
    n_samples=("sample_id","nunique"),
    raw_reads_total=("raw_reads","sum"),
    filtered_reads_total=("filtered_reads","sum"),
    mean_retention_rate=("retention_rate","mean"),
    median_retention_rate=("retention_rate","median"),
)
by_marker.to_csv(outdir / "qc_by_marker.tsv", sep="\t", index=False)
by_scope = df.groupby(["species_scope","sample_type"], as_index=False).agg(
    n_samples=("sample_id","nunique"),
    raw_reads_total=("raw_reads","sum"),
    filtered_reads_total=("filtered_reads","sum"),
    mean_retention_rate=("retention_rate","mean"),
)
by_scope.to_csv(outdir / "qc_by_scope.tsv", sep="\t", index=False)
print("Wrote QC summary tables.")
