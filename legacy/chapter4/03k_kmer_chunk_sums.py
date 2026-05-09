#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-chunk kmer-only analysis (sparse, fast enough, no feature matrix explosion).

Input:
- a chunk sparse kmer file: chunk_XXXX.kmer.tsv.gz
  expected columns include:
    Seq_ID
    kmer_all_json / kmer_head_json / kmer_tail_json
  (auto-detects some common alternatives)

- selected vocab: analysis_results/02_Features/kmervocab/kmer_vocab.tsv
  columns: scope, k, kmer (df/total_count optional)

- Step3A outputs:
  analysis_results/03_DataWeaver/sample_counts/{file_id}.sqlite
  analysis_results/03_DataWeaver/sample_totals.tsv  (only used in reduce)

Output (this chunk):
- analysis_results/03_DataWeaver/kmer_chunk_sums/{chunk}.kmer_sums.tsv.gz
  columns: file_id, scope, k, kmer, sum_count
"""

from __future__ import annotations
import argparse, csv, gzip, json, sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

SQL_VAR_LIMIT = 900

# fast json if available
try:
    import orjson  # type: ignore
    def jloads(x: str):
        return orjson.loads(x)
except Exception:
    def jloads(x: str):
        return json.loads(x)

def detect_cols(fieldnames: List[str]) -> Tuple[str,str,str,str]:
    """
    Returns (seq_id_col, all_col, head_col, tail_col)
    """
    fn = set(fieldnames)
    seq_col = "Seq_ID" if "Seq_ID" in fn else ("seq_id" if "seq_id" in fn else None)
    if not seq_col:
        raise ValueError(f"Cannot find Seq_ID column in {fieldnames}")

    def pick(*cands):
        for c in cands:
            if c in fn:
                return c
        return None

    all_col  = pick("kmer_all_json",  "kmer_all",  "all_json")
    head_col = pick("kmer_head_json", "kmer_head", "head_json")
    tail_col = pick("kmer_tail_json", "kmer_tail", "tail_json")
    if not all_col or not head_col or not tail_col:
        raise ValueError(f"Cannot find kmer json cols; got all={all_col}, head={head_col}, tail={tail_col}; header={fieldnames}")
    return seq_col, all_col, head_col, tail_col

def load_vocab(vocab_tsv: Path) -> Dict[Tuple[str,int], set]:
    sel: Dict[Tuple[str,int], set] = defaultdict(set)
    with vocab_tsv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            scope = row["scope"]
            k = int(row["k"])
            kmer = row["kmer"]
            sel[(scope,k)].add(kmer)
    if not sel:
        raise ValueError(f"Empty vocab: {vocab_tsv}")
    return sel

def fetch_counts(conn: sqlite3.Connection, seq_ids: List[str]) -> Dict[str,int]:
    out: Dict[str,int] = {}
    cur = conn.cursor()
    for i in range(0, len(seq_ids), SQL_VAR_LIMIT):
        batch = seq_ids[i:i+SQL_VAR_LIMIT]
        q = "SELECT seq_id, count FROM counts WHERE seq_id IN (" + ",".join(["?"]*len(batch)) + ")"
        for sid, cnt in cur.execute(q, batch):
            out[sid] = int(cnt)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--chunk_name", required=True, help="chunk_0001")
    ap.add_argument("--kmer_sparse_dir", default=None, help="dir containing chunk_XXXX.kmer.tsv.gz")
    ap.add_argument("--vocab_tsv", default=None, help="default: analysis_results/02_Features/kmervocab/kmer_vocab.tsv")
    args = ap.parse_args()

    project = Path(args.project_dir)
    feat_dir = project / "analysis_results" / "02_Features"
    weaver_dir = project / "analysis_results" / "03_DataWeaver"
    counts_dir = weaver_dir / "sample_counts"
    totals_tsv = weaver_dir / "sample_totals.tsv"

    if not totals_tsv.exists():
        raise FileNotFoundError(f"Missing {totals_tsv}. Run Step3A first.")

    # detect kmer sparse dir
    if args.kmer_sparse_dir:
        kmer_sparse_dir = Path(args.kmer_sparse_dir)
    else:
        # try common candidates
        cands = [
            feat_dir / "kmer_sparse_chunks",
            feat_dir / "kmer_sparse",
            feat_dir / "kmer_chunks",
        ]
        kmer_sparse_dir = None
        for c in cands:
            if c.exists():
                kmer_sparse_dir = c
                break
        if kmer_sparse_dir is None:
            raise FileNotFoundError("Cannot auto-detect kmer_sparse_dir. Pass --kmer_sparse_dir <dir>.")

    vocab_tsv = Path(args.vocab_tsv) if args.vocab_tsv else (feat_dir / "kmervocab" / "kmer_vocab.tsv")
    if not vocab_tsv.exists():
        raise FileNotFoundError(f"Missing vocab: {vocab_tsv} (run 02b/02c vocab selection first)")

    chunk = args.chunk_name
    in_fp = kmer_sparse_dir / f"{chunk}.kmer.tsv.gz"
    if not in_fp.exists():
        raise FileNotFoundError(f"Missing input: {in_fp}")

    out_dir = weaver_dir / "kmer_chunk_sums"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{chunk}.kmer_sums.tsv.gz"
    if out_fp.exists() and out_fp.stat().st_size > 0:
        print(f"[SKIP] exists: {out_fp}")
        return

    selected = load_vocab(vocab_tsv)

    # read sparse kmers for this chunk -> store as per-seq sparse list
    seq_ids: List[str] = []
    seq_kmers: List[List[Tuple[str,int,str]]] = []  # list of (scope,k,kmer) with count for that seq
    with gzip.open(in_fp, "rt", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        if not r.fieldnames:
            raise ValueError(f"Bad header: {in_fp}")
        seq_col, all_col, head_col, tail_col = detect_cols(r.fieldnames)

        for row in r:
            sid = row[seq_col]
            seq_ids.append(sid)

            feats: List[Tuple[str,int,str]] = []

            for scope, col in (("kmer_all", all_col), ("kmer_head", head_col), ("kmer_tail", tail_col)):
                s = row.get(col, "") or "{}"
                try:
                    obj = jloads(s)
                except Exception:
                    obj = {}
                if not isinstance(obj, dict):
                    continue
                for k_str, cmap in obj.items():
                    if not isinstance(cmap, dict):
                        continue
                    k = int(k_str)
                    keep = selected.get((scope,k))
                    if not keep:
                        continue
                    for kmer, cnt in cmap.items():
                        if kmer in keep:
                            c = int(cnt)
                            if c > 0:
                                feats.append((scope, k, kmer, c))
            seq_kmers.append(feats)

    # list all samples from sample_totals.tsv
    sample_ids: List[str] = []
    with totals_tsv.open("r", encoding="utf-8") as f:
        rr = csv.DictReader(f, delimiter="\t")
        for row in rr:
            sample_ids.append(row["file_id"])

    # accumulate sums: (file_id, scope, k, kmer) -> sum(count_sample * kmer_count_in_seq)
    sums = defaultdict(int)

    conns: Dict[str, sqlite3.Connection] = {}
    try:
        for fid in sample_ids:
            dbp = counts_dir / f"{fid}.sqlite"
            if not dbp.exists():
                raise FileNotFoundError(f"Missing {dbp} (run Step3A first)")
            conns[fid] = sqlite3.connect(str(dbp))
            cnt_map = fetch_counts(conns[fid], seq_ids)

            # iterate only seqs present
            for idx, sid in enumerate(seq_ids):
                cseq = cnt_map.get(sid, 0)
                if cseq <= 0:
                    continue
                for scope, k, kmer, kc in seq_kmers[idx]:
                    sums[(fid, scope, k, kmer)] += cseq * kc

        with gzip.open(out_fp, "wt", encoding="utf-8") as fo:
            w = csv.writer(fo, delimiter="\t")
            w.writerow(["file_id","scope","k","kmer","sum_count"])
            for (fid, scope, k, kmer), v in sums.items():
                w.writerow([fid, scope, k, kmer, v])

        print(f"[DONE] {chunk}: rows={len(sums)} -> {out_fp.name}")

    finally:
        for c in conns.values():
            try: c.close()
            except Exception: pass

if __name__ == "__main__":
    main()
