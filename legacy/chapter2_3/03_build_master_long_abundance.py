#!/usr/bin/env python3
from pathlib import Path
import argparse, gzip, hashlib
from collections import Counter
import pandas as pd

def open_maybe_gz(path, mode="rt"):
    return gzip.open(path, mode) if str(path).endswith(".gz") else open(path, mode)

def iter_fastq_sequences(path):
    with open_maybe_gz(path, "rt") as fh:
        while True:
            h = fh.readline()
            if not h: break
            seq = fh.readline().strip().upper()
            fh.readline()
            qual = fh.readline()
            if not qual: break
            if seq: yield seq

def build_counts(path):
    c = Counter()
    for seq in iter_fastq_sequences(path):
        c[seq] += 1
    return c

def seq_id(seq):
    return "SEQ_" + hashlib.md5(seq.encode()).hexdigest()[:16]

ap = argparse.ArgumentParser()
ap.add_argument("--metadata", required=True)
ap.add_argument("--filtered-dir", required=True)
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

meta = pd.read_csv(args.metadata, sep="\t", dtype=str).fillna("")
filtered_dir = Path(args.filtered_dir)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

long_rows, seq_rows, seen = [], [], set()
for _, row in meta.iterrows():
    sample_id = row["sample_id"]
    fq1 = filtered_dir / f"{sample_id}.filtered.fq.gz"
    fq2 = filtered_dir / f"{sample_id}.filtered.fq"
    fq = fq1 if fq1.exists() else fq2
    if not fq.exists():
        print(f"[WARN] missing filtered file for {sample_id}")
        continue
    counts = build_counts(fq)
    total = sum(counts.values())
    for seq, count in counts.items():
        sid = seq_id(seq)
        long_rows.append({
            "sample_id": sample_id,
            "marker": row["marker"],
            "group_name": row["group_name"],
            "sample_type": row["sample_type"],
            "species_scope": row["species_scope"],
            "is_core_analysis": row["is_core_analysis"],
            "sequence_id": sid,
            "sequence": seq,
            "count": count,
            "relative_abundance": 0 if total == 0 else count / total,
        })
        if sid not in seen:
            seq_rows.append({
                "sequence_id": sid,
                "sequence": seq,
                "length": len(seq),
                "gc": round((seq.count("G")+seq.count("C"))/len(seq), 6) if seq else 0.0,
            })
            seen.add(sid)

if not long_rows:
    raise SystemExit("No rows generated.")
long_df = pd.DataFrame(long_rows).sort_values(["sample_id","count"], ascending=[True,False])
seq_df = pd.DataFrame(seq_rows).sort_values(["sequence_id"])
long_df.to_csv(outdir / "master_long_abundance.tsv", sep="\t", index=False)
seq_df.to_csv(outdir / "sequence_catalog.tsv", sep="\t", index=False)
summary = long_df.groupby(["sample_id","marker","group_name","sample_type","species_scope","is_core_analysis"], as_index=False).agg(
    total_reads=("count","sum"),
    n_unique_sequences=("sequence_id","nunique"),
    max_rel_abundance=("relative_abundance","max"),
)
summary.to_csv(outdir / "sample_sequence_summary.tsv", sep="\t", index=False)
with open(outdir / "sequence_catalog.fasta", "w") as f:
    for _, r in seq_df.iterrows():
        f.write(f">{r['sequence_id']}\n{r['sequence']}\n")
print(f"Wrote {outdir / 'master_long_abundance.tsv'}")
