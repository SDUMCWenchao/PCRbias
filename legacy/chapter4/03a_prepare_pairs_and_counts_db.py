#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3A:
1) Read samples_meta.tsv, build yes/no pairs by (species, n_individuals, locus)
2) Build per-sample SQLite DB for counts, to avoid loading huge dict in each Slurm task
3) Write sample_totals.tsv and pairs.tsv

Inputs:
- samples_meta.tsv (columns: file_id, sample_name, species, n_individuals, locus, pcr)
- analysis_results/01_Sequences/{file_id}_stats.tsv

Outputs under analysis_results/03_DataWeaver/:
- sample_counts/{file_id}.sqlite
- sample_totals.tsv
- pairs.tsv
"""

from __future__ import annotations
import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

def read_meta(meta_path: Path) -> List[dict]:
    with meta_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        rows = [row for row in r]
    if not rows:
        raise ValueError(f"Empty meta: {meta_path}")
    need = {"file_id","sample_name","species","n_individuals","locus","pcr"}
    miss = need - set(rows[0].keys())
    if miss:
        raise ValueError(f"meta missing columns: {sorted(miss)}")
    return rows

def ensure_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("""
    CREATE TABLE IF NOT EXISTS counts (
        seq_id TEXT PRIMARY KEY,
        count  INTEGER NOT NULL
    );
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_counts_seq ON counts(seq_id);")
    conn.commit()
    return conn

def build_counts_db(stats_path: Path, db_path: Path) -> Tuple[int,int]:
    """
    Returns (total_reads, unique_seqs)
    """
    conn = ensure_db(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM counts;")
    conn.commit()

    total_reads = 0
    unique = 0
    buf = []
    B = 5000

    with stats_path.open("r", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            sid = row["Seq_ID"]
            cnt = int(row["Count"])
            total_reads += cnt
            unique += 1
            buf.append((sid, cnt))
            if len(buf) >= B:
                cur.executemany("INSERT OR REPLACE INTO counts(seq_id,count) VALUES(?,?)", buf)
                conn.commit()
                buf.clear()
    if buf:
        cur.executemany("INSERT OR REPLACE INTO counts(seq_id,count) VALUES(?,?)", buf)
        conn.commit()
        buf.clear()

    conn.close()
    return total_reads, unique

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--meta_tsv", default=None, help="default: <project_dir>/samples_meta.tsv")
    ap.add_argument("--seq_dir", default=None, help="default: <project_dir>/analysis_results/01_Sequences")
    args = ap.parse_args()

    project = Path(args.project_dir)
    meta_path = Path(args.meta_tsv) if args.meta_tsv else (project / "samples_meta.tsv")
    seq_dir = Path(args.seq_dir) if args.seq_dir else (project / "analysis_results" / "01_Sequences")

    out_dir = project / "analysis_results" / "03_DataWeaver"
    counts_dir = out_dir / "sample_counts"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_dir.mkdir(parents=True, exist_ok=True)

    rows = read_meta(meta_path)

    # ---- build per-sample sqlite + totals
    totals_path = out_dir / "sample_totals.tsv"
    with totals_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["file_id","sample_name","species","n_individuals","locus","pcr","total_reads","unique_sequences","stats_path","sqlite_path"])

        for r in rows:
            fid = r["file_id"]
            stats_path = seq_dir / f"{fid}_stats.tsv"
            if not stats_path.exists():
                raise FileNotFoundError(f"Missing stats: {stats_path}")
            db_path = counts_dir / f"{fid}.sqlite"
            total_reads, uniq = build_counts_db(stats_path, db_path)

            w.writerow([
                fid, r["sample_name"], r["species"], r["n_individuals"], r["locus"], r["pcr"],
                total_reads, uniq, str(stats_path), str(db_path)
            ])
            print(f"[OK] {fid}: total_reads={total_reads} uniq={uniq} -> {db_path.name}")

    # ---- build pairs (group by species+n_individuals+locus)
    groups: Dict[Tuple[str,str,str], Dict[str,List[dict]]] = {}
    for r in rows:
        key = (r["species"], str(r["n_individuals"]), r["locus"])
        groups.setdefault(key, {}).setdefault(r["pcr"], []).append(r)

    pairs_path = out_dir / "pairs.tsv"
    with pairs_path.open("w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","species","n_individuals","locus","yes_file_id","no_file_id","yes_sample_name","no_sample_name"])

        n_pairs = 0
        for (species, nind, locus), d in sorted(groups.items()):
            yes_list = d.get("yes", [])
            no_list  = d.get("no", [])
            if not yes_list or not no_list:
                print(f"[WARN] group missing yes/no: {species} {nind} {locus}")
                continue
            # 若有重复，这里做 all-combinations（最稳）
            for y in yes_list:
                for n in no_list:
                    pair_id = f"{species}__n{nind}__{locus}__{y['file_id']}_vs_{n['file_id']}"
                    w.writerow([pair_id, species, nind, locus, y["file_id"], n["file_id"], y["sample_name"], n["sample_name"]])
                    n_pairs += 1

    print(f"[DONE] totals -> {totals_path}")
    print(f"[DONE] pairs  -> {pairs_path} (n={n_pairs})")

if __name__ == "__main__":
    main()
