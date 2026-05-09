#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

def main():
    ap = argparse.ArgumentParser(description="Create a manageable sequence annotation template for species labeling.")
    ap.add_argument("--abundance", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-global-count", type=int, default=100)
    ap.add_argument("--min-max-rel", type=float, default=0.001)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.abundance, sep="\t")
    ranked = (
        df.groupby(["sequence_id","sequence"], as_index=False)
        .agg(
            total_count=("count","sum"),
            n_samples=("sample_id","nunique"),
            max_rel_abundance=("relative_abundance","max"),
            top_sample=("sample_id", lambda x: x.value_counts().index[0]),
            top_group=("group_name", lambda x: x.value_counts().index[0]),
            markers_seen=("marker", lambda x: ",".join(sorted(set(map(str, x)))))
        )
        .sort_values(["total_count","max_rel_abundance"], ascending=[False,False])
    )

    keep = ranked[
        (ranked["total_count"] >= args.min_global_count) |
        (ranked["max_rel_abundance"] >= args.min_max_rel)
    ].copy()

    keep["species_label"] = ""
    keep["annotation_status"] = "pending"
    keep["is_target_candidate"] = ""
    keep["notes"] = "Fill species_label with the exact label used for downstream expected_design files"

    keep.to_csv(outdir / "sequence_annotation_template.tsv", sep="\t", index=False)

    with (outdir / "sequence_annotation_template.fasta").open("w") as f:
        for _, r in keep.iterrows():
            f.write(f">{r['sequence_id']}|total_count={r['total_count']}|max_rel={r['max_rel_abundance']:.6f}\n{r['sequence']}\n")

    print(f"Wrote {outdir / 'sequence_annotation_template.tsv'}")
    print(f"Wrote {outdir / 'sequence_annotation_template.fasta'}")

if __name__ == "__main__":
    main()
