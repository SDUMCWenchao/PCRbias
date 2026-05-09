#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

BLAST_COLS = ["qseqid","sseqid","pident","length","qcovs","evalue","bitscore","sscinames","stitle"]

def parse_species_from_sseqid(sseqid: str, sep: str = "|", field_idx: int = 0) -> str:
    s = str(sseqid)
    parts = s.split(sep)
    return parts[field_idx].strip() if field_idx < len(parts) else s.strip()

def classify_hit(pident, qcovs, high_pid, high_qcov, med_pid, med_qcov, low_pid, low_qcov):
    pident = float(pident)
    qcovs = float(qcovs)
    if pident >= high_pid and qcovs >= high_qcov:
        return "high_confidence"
    if pident >= med_pid and qcovs >= med_qcov:
        return "medium_confidence"
    if pident >= low_pid and qcovs >= low_qcov:
        return "low_confidence"
    return "unresolved"

def main():
    ap = argparse.ArgumentParser(description="Collect best BLAST hit per query and build marker-aware annotation table.")
    ap.add_argument("--blast-dir", required=True)
    ap.add_argument("--query-summary", required=True, help="annotation_queries_summary.tsv from script 11")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sseqid-sep", default="|")
    ap.add_argument("--sseqid-field-idx", type=int, default=0)
    ap.add_argument("--high-pid", type=float, default=99.0)
    ap.add_argument("--high-qcov", type=float, default=95.0)
    ap.add_argument("--med-pid", type=float, default=97.0)
    ap.add_argument("--med-qcov", type=float, default=90.0)
    ap.add_argument("--low-pid", type=float, default=95.0)
    ap.add_argument("--low-qcov", type=float, default=80.0)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qs = pd.read_csv(args.query_summary, sep="\t")
    all_hits = []

    for marker in sorted(qs["marker"].astype(str).unique()):
        f = Path(args.blast_dir) / f"{marker}.blast.tsv"
        if not f.exists() or f.stat().st_size == 0:
            continue
        df = pd.read_csv(f, sep="\t", header=None, names=BLAST_COLS)
        if df.empty:
            continue
        df["marker"] = marker
        all_hits.append(df)

    if all_hits:
        hits = pd.concat(all_hits, ignore_index=True)
        hits = hits.sort_values(
            ["marker","qseqid","bitscore","pident","qcovs","length"],
            ascending=[True,True,False,False,False,False]
        )
        best = hits.groupby(["marker","qseqid"], as_index=False).first()
        best["species_label"] = best["sseqid"].apply(parse_species_from_sseqid, sep=args.sseqid_sep, field_idx=args.sseqid_field_idx)
        best["annotation_status"] = best.apply(
            lambda r: classify_hit(r["pident"], r["qcovs"], args.high_pid, args.high_qcov, args.med_pid, args.med_qcov, args.low_pid, args.low_qcov),
            axis=1,
        )
        best["source_method"] = "blastn_local"
    else:
        best = pd.DataFrame(columns=["marker","qseqid","species_label","annotation_status","source_method"])

    merged = qs.merge(best, left_on=["marker","sequence_id"], right_on=["marker","qseqid"], how="left")
    merged["species_label"] = merged["species_label"].fillna("UNANNOTATED")
    merged["annotation_status"] = merged["annotation_status"].fillna("unresolved")
    merged["source_method"] = merged["source_method"].fillna("blastn_local")
    merged["is_target_candidate"] = ""

    merged["notes"] = merged.apply(
        lambda r: "" if pd.notna(r.get("sseqid", None)) else "No BLAST hit under current database/threshold",
        axis=1
    )

    ann = merged[[
        "marker","sequence_id","sequence","species_label","annotation_status","is_target_candidate","source_method",
        "pident","qcovs","bitscore","sseqid","sscinames","stitle",
        "total_count","n_samples","max_rel_abundance","top_sample","top_group","notes"
    ]].copy()

    ann.to_csv(outdir / "sequence_annotation_full.tsv", sep="\t", index=False)

    unresolved = ann[ann["annotation_status"].isin(["unresolved"])].copy()
    unresolved.to_csv(outdir / "sequence_annotation_unresolved.tsv", sep="\t", index=False)
    with (outdir / "sequence_annotation_unresolved.fasta").open("w") as f:
        for _, r in unresolved.iterrows():
            f.write(
                f">{r['sequence_id']}|marker={r['marker']}|total_count={r['total_count']}|max_rel={r['max_rel_abundance']:.6f}\n"
                f"{r['sequence']}\n"
            )

    summary = ann.groupby(["marker","annotation_status"], as_index=False).agg(
        n_sequences=("sequence_id","nunique"),
        total_count=("total_count","sum")
    )
    summary.to_csv(outdir / "sequence_annotation_status_summary.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'sequence_annotation_full.tsv'}")
    print(f"Wrote {outdir / 'sequence_annotation_unresolved.tsv'}")
    print(f"Wrote {outdir / 'sequence_annotation_unresolved.fasta'}")
    print(f"Wrote {outdir / 'sequence_annotation_status_summary.tsv'}")

if __name__ == "__main__":
    main()
