#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 3B worker:
For a given chunk (chunk_XXXX), read core feature chunk and generate training rows for all pairs:
Output: analysis_results/03_DataWeaver/training_chunks/<chunk>.train.tsv.gz

Each output row:
pair_id, yes_file_id, no_file_id, Seq_ID,
count_yes, count_no, total_yes, total_no,
rel_yes, rel_no, log2fc,
+ all core feature columns (excluding Seq_ID)

Label:
log2fc = log2( ((count_yes + 0.5)/(total_yes + 0.5)) / ((count_no + 0.5)/(total_no + 0.5)) )
"""

from __future__ import annotations
import argparse, csv, gzip, math, sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

SQL_VAR_LIMIT = 900  # avoid "too many SQL variables" (default 999)

def read_pairs(pairs_path: Path) -> List[dict]:
    with pairs_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [row for row in r]

def read_totals(totals_path: Path) -> Dict[str,int]:
    d = {}
    with totals_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            d[row["file_id"]] = int(row["total_reads"])
    return d

def fetch_counts(conn: sqlite3.Connection, seq_ids: List[str]) -> Dict[str,int]:
    """
    Fetch counts for many seq_ids, batching to avoid SQLite variable limit.
    """
    out: Dict[str,int] = {}
    cur = conn.cursor()
    for i in range(0, len(seq_ids), SQL_VAR_LIMIT):
        batch = seq_ids[i:i+SQL_VAR_LIMIT]
        q = "SELECT seq_id, count FROM counts WHERE seq_id IN (" + ",".join(["?"]*len(batch)) + ")"
        for sid, cnt in cur.execute(q, batch):
            out[sid] = int(cnt)
    return out

def read_core_chunk(core_path: Path) -> Tuple[List[str], List[dict]]:
    rows = []
    with gzip.open(core_path, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        fieldnames = r.fieldnames
        if not fieldnames or "Seq_ID" not in fieldnames:
            raise ValueError(f"Bad core file header: {core_path}")
        for row in r:
            rows.append(row)
    return fieldnames, rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--chunk_name", required=True, help="e.g., chunk_0001")
    ap.add_argument("--feature_kind", default="core", choices=["core"], help="currently only core (recommended)")
    args = ap.parse_args()

    project = Path(args.project_dir)
    chunk = args.chunk_name

    feat_dir = project / "analysis_results" / "02_Features"
    core_path = feat_dir / "core_chunks" / f"{chunk}.core.tsv.gz"
    if not core_path.exists():
        raise FileNotFoundError(f"Missing core chunk: {core_path}")

    out_dir = project / "analysis_results" / "03_DataWeaver"
    pairs_path = out_dir / "pairs.tsv"
    totals_path = out_dir / "sample_totals.tsv"
    counts_dir = out_dir / "sample_counts"
    train_dir = out_dir / "training_chunks"
    train_dir.mkdir(parents=True, exist_ok=True)

    out_path = train_dir / f"{chunk}.train.tsv.gz"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] exists: {out_path}")
        return

    pairs = read_pairs(pairs_path)
    totals = read_totals(totals_path)

    # Load core chunk rows
    fieldnames, rows = read_core_chunk(core_path)
    feat_cols = [c for c in fieldnames if c != "Seq_ID"]
    seq_ids = [r["Seq_ID"] for r in rows]

    # Determine all samples involved in these pairs (so we can fetch counts once per sample)
    sample_ids = sorted(set([p["yes_file_id"] for p in pairs] + [p["no_file_id"] for p in pairs]))

    # Open sqlite connections + fetch counts for this chunk
    counts_by_sample: Dict[str, Dict[str,int]] = {}
    conns: Dict[str, sqlite3.Connection] = {}
    try:
        for sid in sample_ids:
            db_path = counts_dir / f"{sid}.sqlite"
            if not db_path.exists():
                raise FileNotFoundError(f"Missing sample sqlite: {db_path} (run Step3A first)")
            conns[sid] = sqlite3.connect(str(db_path))
            counts_by_sample[sid] = fetch_counts(conns[sid], seq_ids)

        # Write output
        out_cols = [
            "pair_id","yes_file_id","no_file_id","Seq_ID",
            "count_yes","count_no","total_yes","total_no",
            "rel_yes","rel_no","log2fc"
        ] + feat_cols

        with gzip.open(out_path, "wt", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=out_cols, delimiter="\t")
            w.writeheader()

            wrote = 0
            for p in pairs:
                pid = p["pair_id"]
                yid = p["yes_file_id"]
                nid = p["no_file_id"]
                ty = totals[yid]
                tn = totals[nid]
                cy_map = counts_by_sample[yid]
                cn_map = counts_by_sample[nid]

                for r in rows:
                    sid = r["Seq_ID"]
                    cy = cy_map.get(sid, 0)
                    cn = cn_map.get(sid, 0)
                    if cy == 0 and cn == 0:
                        continue

                    # rel + log2fc with pseudocount 0.5 (stable)
                    rel_y = (cy + 0.5) / (ty + 0.5)
                    rel_n = (cn + 0.5) / (tn + 0.5)
                    log2fc = math.log2(rel_y / rel_n) if (rel_y > 0 and rel_n > 0) else 0.0

                    out = {
                        "pair_id": pid,
                        "yes_file_id": yid,
                        "no_file_id": nid,
                        "Seq_ID": sid,
                        "count_yes": cy,
                        "count_no": cn,
                        "total_yes": ty,
                        "total_no": tn,
                        "rel_yes": f"{rel_y:.12g}",
                        "rel_no": f"{rel_n:.12g}",
                        "log2fc": f"{log2fc:.12g}",
                    }
                    # add features (keep as text; modeling step can cast)
                    for c in feat_cols:
                        out[c] = r.get(c, "")
                    w.writerow(out)
                    wrote += 1

        print(f"[DONE] {chunk}: wrote={wrote} -> {out_path.name}")

    finally:
        for c in conns.values():
            try: c.close()
            except Exception: pass

if __name__ == "__main__":
    main()
