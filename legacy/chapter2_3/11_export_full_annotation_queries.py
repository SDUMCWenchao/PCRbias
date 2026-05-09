#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Export full marker-split query FASTA and summary tables for annotation.")
    ap.add_argument("--abundance", required=True, help="Use rel_ge_0.001.filtered.tsv or another filtered abundance table")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--core-only", action="store_true", help="Only export is_core_analysis == yes")
    ap.add_argument("--min-global-count", type=int, default=1, help="Keep sequences with total count >= this value")
    ap.add_argument("--min-max-rel", type=float, default=0.0, help="Keep sequences with max relative abundance >= this value")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ab = pd.read_csv(args.abundance, sep="\t")
    if args.core_only:
        ab = ab[ab["is_core_analysis"].astype(str).str.lower() == "yes"].copy()

    seqsum = (
        ab.groupby(["marker","sequence_id","sequence"], as_index=False)
        .agg(
            total_count=("count","sum"),
            n_samples=("sample_id","nunique"),
            max_rel_abundance=("relative_abundance","max"),
            top_sample=("sample_id", lambda x: x.value_counts().index[0]),
            top_group=("group_name", lambda x: x.value_counts().index[0]),
            sample_types=("sample_type", lambda x: ",".join(sorted(set(map(str, x)))))
        )
        .sort_values(["marker","total_count","max_rel_abundance"], ascending=[True,False,False])
    )

    keep = seqsum[
        (seqsum["total_count"] >= args.min_global_count) |
        (seqsum["max_rel_abundance"] >= args.min_max_rel)
    ].copy()

    keep.to_csv(outdir / "annotation_queries_summary.tsv", sep="\t", index=False)

    for marker, sub in keep.groupby("marker", sort=True):
        sub.to_csv(outdir / f"annotation_queries_{marker}.tsv", sep="\t", index=False)
        with (outdir / f"annotation_queries_{marker}.fasta").open("w") as f:
            for _, r in sub.iterrows():
                f.write(
                    f">{r['sequence_id']}|marker={marker}|total_count={r['total_count']}|max_rel={r['max_rel_abundance']:.6f}|top_group={r['top_group']}\n"
                    f"{r['sequence']}\n"
                )

    print(f"Wrote {outdir / 'annotation_queries_summary.tsv'}")
    for marker in sorted(keep["marker"].dropna().astype(str).unique()):
        print(f"Wrote {outdir / f'annotation_queries_{marker}.tsv'}")
        print(f"Wrote {outdir / f'annotation_queries_{marker}.fasta'}")

if __name__ == "__main__":
    main()
