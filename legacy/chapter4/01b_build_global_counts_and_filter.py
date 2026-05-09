#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build global counts across all samples from *_stats.tsv (Step1 outputs),
then filter ALL_UNIQUE_SEQUENCES.fasta by global_total_count >= min_count.

Outputs (under analysis_results/01_Sequences/):
  - global_counts.sqlite
  - global_counts_summary.txt
  - ALL_UNIQUE_SEQUENCES.countge{min_count}.fasta
"""

from __future__ import annotations
import argparse
import csv
import gzip
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple

def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS counts (
        seq_id TEXT PRIMARY KEY,
        total_count INTEGER NOT NULL,
        n_samples INTEGER NOT NULL
    );
    """)
    conn.commit()
    return conn

def iter_stats_rows(stats_path: Path) -> Iterable[Tuple[str,int]]:
    with stats_path.open("r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            sid = row["Seq_ID"]
            cnt = int(row["Count"])
            yield sid, cnt

def fasta_iter(path: Path):
    header = None
    seq_parts = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if header is not None:
            yield header, "".join(seq_parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--seq_dir", default=None, help="analysis_results/01_Sequences (default under project)")
    ap.add_argument("--min_count", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    project = Path(args.project_dir)
    seq_dir = Path(args.seq_dir) if args.seq_dir else (project / "analysis_results" / "01_Sequences")
    seq_dir.mkdir(parents=True, exist_ok=True)

    db_path = seq_dir / "global_counts.sqlite"
    out_fa = seq_dir / f"ALL_UNIQUE_SEQUENCES.countge{args.min_count}.fasta"
    summary_txt = seq_dir / "global_counts_summary.txt"

    if out_fa.exists() and (not args.force):
        print(f"[SKIP] {out_fa} exists. Use --force to rebuild.")
        return

    stats_files = sorted(seq_dir.glob("*_stats.tsv"))
    if not stats_files:
        raise FileNotFoundError(f"No *_stats.tsv found in {seq_dir} (run Step1 first)")

    all_fa = seq_dir / "ALL_UNIQUE_SEQUENCES.fasta"
    if not all_fa.exists():
        raise FileNotFoundError(f"Missing {all_fa} (run Step1 first)")

    conn = ensure_db(db_path)
    cur = conn.cursor()

    # upsert statement
    upsert = """
    INSERT INTO counts(seq_id, total_count, n_samples)
    VALUES(?, ?, 1)
    ON CONFLICT(seq_id) DO UPDATE SET
        total_count = total_count + excluded.total_count,
        n_samples = n_samples + 1;
    """

    print(f"[INFO] Building global counts from {len(stats_files)} sample stats...")
    batch = []
    BATCH_N = 5000
    for fp in stats_files:
        n = 0
        for sid, cnt in iter_stats_rows(fp):
            batch.append((sid, cnt))
            n += 1
            if len(batch) >= BATCH_N:
                cur.executemany(upsert, batch)
                conn.commit()
                batch.clear()
        if batch:
            cur.executemany(upsert, batch)
            conn.commit()
            batch.clear()
        print(f"[OK] {fp.name}: {n} unique seqs processed")

    # How many seq_ids qualify?
    cur.execute("SELECT COUNT(*) FROM counts;")
    n_total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM counts WHERE total_count >= ?;", (args.min_count,))
    n_keep = cur.fetchone()[0]

    # Build a set of kept seq_ids (fast membership while filtering fasta)
    # If memory is ever a concern, we can replace this with a sorted-merge approach,
    # but usually count>=2 reduces size a lot.
    print(f"[INFO] Loading kept Seq_IDs (total_count>={args.min_count}) into memory...")
    kept = set()
    for (sid,) in cur.execute("SELECT seq_id FROM counts WHERE total_count >= ?;", (args.min_count,)):
        kept.add(sid)
    print(f"[INFO] kept set size = {len(kept)}")

    # Filter fasta
    print(f"[INFO] Filtering fasta: {all_fa} -> {out_fa}")
    kept_written = 0
    with out_fa.open("w", encoding="utf-8") as fo:
        for sid, seq in fasta_iter(all_fa):
            if sid in kept:
                fo.write(f">{sid}\n{seq}\n")
                kept_written += 1

    with summary_txt.open("w", encoding="utf-8") as fo:
        fo.write(f"global_counts.sqlite: {db_path}\n")
        fo.write(f"min_count: {args.min_count}\n")
        fo.write(f"total_seq_ids_in_counts: {n_total}\n")
        fo.write(f"qualified_seq_ids: {n_keep}\n")
        fo.write(f"written_to_fasta: {kept_written}\n")

    conn.close()

    print("[DONE]")
    print(f"  DB:      {db_path}")
    print(f"  FASTA:   {out_fa} (n={kept_written})")
    print(f"  SUMMARY: {summary_txt}")

if __name__ == "__main__":
    main()
