#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 1: Preprocess single-end FASTQ samples from manifest.tsv
- For each sample: count identical sequences -> {file_id}_stats.tsv
- Build a global SQLite database of unique sequences: global_sequences.sqlite
- Export ALL_UNIQUE_SEQUENCES.fasta (Seq_ID -> Sequence) from the SQLite DB
- Write samples_summary.tsv (reads, unique seqs, length stats, N stats)

Assumptions:
- All inputs are single-end FASTQ (.fq/.fastq, optionally .gz)
- manifest.tsv has columns: file_id, sample_name, species, n_individuals, locus, pcr, fastq_path, status
"""

from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def open_text_maybe_gz(p: Path):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return p.open("r", encoding="utf-8", errors="replace")


def fastq_sequences(path: Path, validate: bool = False) -> Iterator[str]:
    """
    Yield sequences from a FASTQ file (single-end).
    Reads 4 lines per record; yields the sequence line (uppercased).
    """
    with open_text_maybe_gz(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline()
            p = f.readline()
            q = f.readline()
            if not (s and p and q):
                break
            if validate:
                if not h.startswith("@") or not p.startswith("+"):
                    raise ValueError(f"FASTQ format error in {path}: bad header/plus line")
                if len(s.rstrip("\n")) != len(q.rstrip("\n")):
                    raise ValueError(f"FASTQ format error in {path}: seq/qual length mismatch")
            yield s.strip().upper()


def md5_seq(seq: str) -> str:
    return hashlib.md5(seq.encode("utf-8")).hexdigest()


def ensure_sqlite(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sequences (
            seq_id TEXT PRIMARY KEY,
            sequence TEXT NOT NULL,
            length INTEGER NOT NULL
        );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sequences_len ON sequences(length);")
    conn.commit()
    return conn


def read_manifest(manifest_path: Path) -> List[dict]:
    with manifest_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader]
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")
    required = {"file_id", "fastq_path", "status"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return rows


def write_sample_stats(
    out_path: Path,
    meta: dict,
    counter: Counter,
    total_reads: int,
    top_fracs: List[float],
    min_count_tag: int = 2,
):
    """
    Write per-sample sequence stats table:
    Seq_ID, Sequence, Length, Count, Rel_Abund, Tag
    Sorted by Count desc.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # compute cutoff ranks for "topX%" tags (by abundance rank)
    uniq = len(counter)
    cutoffs = {}
    for frac in top_fracs:
        k = max(1, int(round(uniq * frac)))
        cutoffs[frac] = k

    items = counter.most_common()
    fieldnames = [
        "file_id", "sample_name", "species", "n_individuals", "locus", "pcr",
        "Seq_ID", "Sequence", "Length", "Count", "Rel_Abund", "Tag"
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        for rank, (seq, cnt) in enumerate(items, start=1):
            sid = md5_seq(seq)
            rel = cnt / total_reads if total_reads > 0 else 0.0

            tags = []
            if cnt >= min_count_tag:
                tags.append(f"count{min_count_tag}")
            for frac, cutoff in cutoffs.items():
                if rank <= cutoff:
                    # e.g. top1pct, top0.5pct
                    pct = frac * 100
                    if abs(pct - round(pct)) < 1e-9:
                        tags.append(f"top{int(round(pct))}pct")
                    else:
                        tags.append(f"top{pct:g}pct")

            w.writerow({
                "file_id": meta.get("file_id", ""),
                "sample_name": meta.get("sample_name", ""),
                "species": meta.get("species", ""),
                "n_individuals": meta.get("n_individuals", ""),
                "locus": meta.get("locus", ""),
                "pcr": meta.get("pcr", ""),
                "Seq_ID": sid,
                "Sequence": seq,
                "Length": len(seq),
                "Count": cnt,
                "Rel_Abund": f"{rel:.12g}",
                "Tag": ",".join(tags) if tags else "",
            })


def sqlite_insert_sequences(conn: sqlite3.Connection, seqs: Iterable[str], chunk: int = 5000):
    """
    Insert unique sequences into SQLite as (seq_id, sequence, length) with INSERT OR IGNORE.
    seqs should be unique at caller side to reduce work.
    """
    buf = []
    cur = conn.cursor()
    for seq in seqs:
        sid = md5_seq(seq)
        buf.append((sid, seq, len(seq)))
        if len(buf) >= chunk:
            cur.executemany("INSERT OR IGNORE INTO sequences(seq_id, sequence, length) VALUES(?,?,?)", buf)
            conn.commit()
            buf.clear()
    if buf:
        cur.executemany("INSERT OR IGNORE INTO sequences(seq_id, sequence, length) VALUES(?,?,?)", buf)
        conn.commit()


def export_all_unique_fasta(conn: sqlite3.Connection, out_fasta: Path, out_tsv: Path | None = None):
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    cur = conn.cursor()
    cur.execute("SELECT seq_id, sequence FROM sequences;")

    with out_fasta.open("w", encoding="utf-8") as fo:
        if out_tsv:
            out_tsv.parent.mkdir(parents=True, exist_ok=True)
            tsv_f = gzip.open(out_tsv, "wt", encoding="utf-8") if str(out_tsv).endswith(".gz") else out_tsv.open("w", encoding="utf-8")
            tsv_f.write("Seq_ID\tLength\tSequence\n")
        else:
            tsv_f = None

        n = 0
        for sid, seq in cur:
            fo.write(f">{sid}\n{seq}\n")
            if tsv_f:
                tsv_f.write(f"{sid}\t{len(seq)}\t{seq}\n")
            n += 1

        if tsv_f:
            tsv_f.close()

    return n


def main():
    ap = argparse.ArgumentParser(description="Step 1: preprocess single-end FASTQ -> per-sample counts + global unique fasta")
    ap.add_argument("--project_dir", default=None, help="Project root. Default: parent of this script directory.")
    ap.add_argument("--manifest", default=None, help="Path to manifest.tsv. Default: <project_dir>/analysis_results/00_manifest/manifest.tsv")
    ap.add_argument("--out_dir", default=None, help="Output dir. Default: <project_dir>/analysis_results/01_Sequences")
    ap.add_argument("--db", default=None, help="SQLite path. Default: <out_dir>/global_sequences.sqlite")
    ap.add_argument("--validate_fastq", action="store_true", help="Strict FASTQ validation while reading.")
    ap.add_argument("--max_reads", type=int, default=0, help="For testing: stop after reading N reads per sample (0 = all).")
    ap.add_argument("--top_fracs", default="0.01,0.005", help="Comma-separated fractions for top tags, e.g. 0.01,0.005")
    ap.add_argument("--min_count_tag", type=int, default=2, help="Add tag countK when Count>=K.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).resolve() if args.project_dir else script_dir.parent

    manifest_path = Path(args.manifest).resolve() if args.manifest else (project_dir / "analysis_results" / "00_manifest" / "manifest.tsv")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_dir / "analysis_results" / "01_Sequences")
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db).resolve() if args.db else (out_dir / "global_sequences.sqlite")

    rows = read_manifest(manifest_path)
    top_fracs = []
    for x in args.top_fracs.split(","):
        x = x.strip()
        if not x:
            continue
        top_fracs.append(float(x))

    conn = ensure_sqlite(db_path)

    summary_path = out_dir / "samples_summary.tsv"
    seq_dir = out_dir  # per-sample stats in the same folder
    all_unique_fasta = out_dir / "ALL_UNIQUE_SEQUENCES.fasta"
    all_unique_tsvgz = out_dir / "ALL_UNIQUE_SEQUENCES.tsv.gz"

    # Write summary
    sum_fields = [
        "file_id", "sample_name", "species", "n_individuals", "locus", "pcr",
        "status",
        "fastq_path",
        "total_reads", "unique_sequences",
        "min_len", "mean_len", "max_len",
        "n_reads_with_N", "frac_reads_with_N"
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as fo_sum:
        sw = csv.DictWriter(fo_sum, fieldnames=sum_fields, delimiter="\t")
        sw.writeheader()

        n_ok = 0
        n_skip = 0

        for meta in rows:
            status = (meta.get("status") or "").strip()
            fid = (meta.get("file_id") or "").strip()
            fastq_path = (meta.get("fastq_path") or "").strip()

            if status == "MISSING" or not fastq_path:
                n_skip += 1
                sw.writerow({
                    **{k: meta.get(k, "") for k in ["file_id", "sample_name", "species", "n_individuals", "locus", "pcr"]},
                    "status": "MISSING",
                    "fastq_path": fastq_path,
                    "total_reads": 0, "unique_sequences": 0,
                    "min_len": "", "mean_len": "", "max_len": "",
                    "n_reads_with_N": 0, "frac_reads_with_N": "",
                })
                continue

            fq = Path(fastq_path)
            if not fq.exists():
                n_skip += 1
                sw.writerow({
                    **{k: meta.get(k, "") for k in ["file_id", "sample_name", "species", "n_individuals", "locus", "pcr"]},
                    "status": "PATH_NOT_FOUND",
                    "fastq_path": fastq_path,
                    "total_reads": 0, "unique_sequences": 0,
                    "min_len": "", "mean_len": "", "max_len": "",
                    "n_reads_with_N": 0, "frac_reads_with_N": "",
                })
                continue

            counter = Counter()
            total = 0
            lens_sum = 0
            min_len = None
            max_len = 0
            n_with_N = 0

            for seq in fastq_sequences(fq, validate=args.validate_fastq):
                total += 1
                if args.max_reads and total >= args.max_reads:
                    # include this read then stop
                    pass

                counter[seq] += 1
                L = len(seq)
                lens_sum += L
                min_len = L if min_len is None else min(min_len, L)
                max_len = max(max_len, L)
                if "N" in seq:
                    n_with_N += 1

                if args.max_reads and total >= args.max_reads:
                    break

            uniq = len(counter)
            mean_len = (lens_sum / total) if total else 0.0
            fracN = (n_with_N / total) if total else 0.0

            # write per-sample stats
            stats_path = seq_dir / f"{fid}_stats.tsv"
            write_sample_stats(
                out_path=stats_path,
                meta=meta,
                counter=counter,
                total_reads=total,
                top_fracs=top_fracs,
                min_count_tag=args.min_count_tag,
            )

            # insert unique sequences into SQLite
            sqlite_insert_sequences(conn, counter.keys())

            sw.writerow({
                **{k: meta.get(k, "") for k in ["file_id", "sample_name", "species", "n_individuals", "locus", "pcr"]},
                "status": "OK",
                "fastq_path": fastq_path,
                "total_reads": total,
                "unique_sequences": uniq,
                "min_len": min_len if min_len is not None else "",
                "mean_len": f"{mean_len:.6f}",
                "max_len": max_len,
                "n_reads_with_N": n_with_N,
                "frac_reads_with_N": f"{fracN:.6g}",
            })

            n_ok += 1
            print(f"[OK] {fid}: reads={total} unique={uniq} -> {stats_path.name}")

    # export global unique fasta + tsv.gz
    n_global = export_all_unique_fasta(conn, all_unique_fasta, out_tsv=all_unique_tsvgz)
    conn.close()

    print("\n[DONE]")
    print(f"  per-sample stats dir: {out_dir}")
    print(f"  samples summary:      {summary_path}")
    print(f"  global sqlite:        {db_path}")
    print(f"  ALL_UNIQUE fasta:     {all_unique_fasta} (n={n_global})")
    print(f"  ALL_UNIQUE tsv.gz:    {all_unique_tsvgz}")


if __name__ == "__main__":
    main()
