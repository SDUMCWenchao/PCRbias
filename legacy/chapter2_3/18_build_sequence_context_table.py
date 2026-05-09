#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

DEFAULT_THRESHOLD_LABELS = ["rel_ge_0.005", "rel_ge_0.01"]

def split_regions(seq: str) -> dict:
    seq = str(seq)
    L = len(seq)
    if L < 61:
        raise ValueError(f"Sequence too short for fixed head/tail split: length={L}")
    head = seq[:30]
    tail = seq[-30:]
    middle = seq[30:-30]
    m = len(middle)
    base = m // 3
    rem = m % 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]
    a = sizes[0]
    b = sizes[0] + sizes[1]
    mid1 = middle[:a]
    mid2 = middle[a:b]
    mid3 = middle[b:]
    return {
        "length": L,
        "head": head,
        "mid1": mid1,
        "mid2": mid2,
        "mid3": mid3,
        "tail": tail,
    }

def main():
    ap = argparse.ArgumentParser(description="Build sequence context table with marker and fixed regional segmentation.")
    ap.add_argument("--renorm-threshold-dir", required=True, help="Directory with *.annotated.renorm.tsv")
    ap.add_argument("--sequence-catalog", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold-labels", nargs="*", default=DEFAULT_THRESHOLD_LABELS)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seqcat = pd.read_csv(args.sequence_catalog, sep="\t", dtype=str).fillna("")
    seqcat = seqcat[["sequence_id", "sequence"]].drop_duplicates("sequence_id")

    map_frames = []
    for label in args.threshold_labels:
        fp = Path(args.renorm_threshold_dir) / f"{label}.annotated.renorm.tsv"
        if not fp.exists():
            print(f"[WARN] missing {fp}")
            continue
        df = pd.read_csv(fp, sep="\t", dtype=str).fillna("")
        # compat
        if "marker" not in df.columns:
            if "marker_x" in df.columns:
                df["marker"] = df["marker_x"]
            elif "marker_y" in df.columns:
                df["marker"] = df["marker_y"]
        keep = [c for c in ["sequence_id","marker","species_label","sample_type"] if c in df.columns]
        map_frames.append(df[keep].drop_duplicates())

    if not map_frames:
        raise SystemExit("No threshold tables found for sequence-marker mapping.")

    seqmap = pd.concat(map_frames, ignore_index=True).drop_duplicates(["sequence_id","marker"])
    seqmap = (
        seqmap.groupby("sequence_id", as_index=False)
        .agg(
            marker=("marker", lambda x: ",".join(sorted(set(map(str, x))))),
            sample_types=("sample_type", lambda x: ",".join(sorted(set(map(str, x))))),
            species_labels=("species_label", lambda x: ",".join(sorted(set(map(str, x))))[:1000]),
        )
    )

    merged = seqcat.merge(seqmap, on="sequence_id", how="left")
    merged["marker"] = merged["marker"].replace("", "UNKNOWN").fillna("UNKNOWN")

    region_rows = []
    for _, r in merged.iterrows():
        seq = str(r["sequence"]).upper()
        parts = split_regions(seq)
        region_rows.append({
            "sequence_id": r["sequence_id"],
            "marker": r["marker"],
            "sample_types_seen": r.get("sample_types", ""),
            "species_labels_seen": r.get("species_labels", ""),
            "sequence": seq,
            "length": parts["length"],
            "head": parts["head"],
            "mid1": parts["mid1"],
            "mid2": parts["mid2"],
            "mid3": parts["mid3"],
            "tail": parts["tail"],
        })

    out = pd.DataFrame(region_rows)
    out.to_csv(outdir / "sequence_regions.tsv", sep="\t", index=False)

    region_lengths = out[["sequence_id","length"]].copy()
    for col in ["head","mid1","mid2","mid3","tail"]:
        region_lengths[f"{col}_len"] = out[col].str.len()
    region_lengths.to_csv(outdir / "sequence_region_lengths.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'sequence_regions.tsv'}")
    print(f"Wrote {outdir / 'sequence_region_lengths.tsv'}")

if __name__ == "__main__":
    main()
