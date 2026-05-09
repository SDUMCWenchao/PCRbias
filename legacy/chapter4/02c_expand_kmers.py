#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 2C: Expand sparse k-mers to dense columns using selected vocab,
and merge with core features.

Inputs:
  core_chunks/*.core.tsv.gz
  kmer_sparse_chunks/*.kmer.tsv.gz
  kmervocab/kmer_vocab.tsv

Outputs:
  final_chunks/<chunk>.final.tsv.gz  (NO Sequence column; contains selected k-mer cols)

Column naming for kmers:
  feat_<scope>_k<k>_<kmer>
e.g.
  feat_kmer_all_k6_AACGTT
  feat_kmer_head_k8_GGAT...

Note:
- Values are raw counts (you can later normalize by length if desired).
"""

from __future__ import annotations
import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List, Tuple

def read_tsv_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            yield r

def write_tsv_gz(path: Path, fieldnames: List[str], rows: List[Dict[str,object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(k, "")) for k in fieldnames) + "\n")

def load_vocab(vocab_path: Path) -> Dict[Tuple[str,int,str], str]:
    """
    Return mapping key=(scope,k,kmer) -> column_name
    """
    m = {}
    with vocab_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            scope = r["scope"]
            k = int(r["k"])
            kmer = r["kmer"]
            col = f"feat_{scope}_k{k}_{kmer}"
            m[(scope,k,kmer)] = col
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--feature_dir", default=None)
    ap.add_argument("--chunk_name", default=None, help="If set, only expand this chunk (e.g., chunk_0001)")
    args = ap.parse_args()

    project = Path(args.project_dir)
    feature_dir = Path(args.feature_dir) if args.feature_dir else (project / "analysis_results" / "02_Features")

    core_dir = feature_dir / "core_chunks"
    kmer_dir = feature_dir / "kmer_sparse_chunks"
    vocab_path = feature_dir / "kmervocab" / "kmer_vocab.tsv"
    out_dir = feature_dir / "final_chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab = load_vocab(vocab_path)
    all_kmer_cols = sorted(set(vocab.values()))

    # choose chunks
    core_files = sorted(core_dir.glob("*.core.tsv.gz"))
    if args.chunk_name:
        core_files = [p for p in core_files if p.name.startswith(args.chunk_name + ".")]

    if not core_files:
        raise FileNotFoundError(f"No core chunks found in {core_dir} (chunk_name={args.chunk_name})")

    for core_fp in core_files:
        chunk = core_fp.name.split(".")[0]  # chunk_0001
        kmer_fp = kmer_dir / f"{chunk}.kmer.tsv.gz"
        if not kmer_fp.exists():
            raise FileNotFoundError(f"Missing kmer sparse file for {chunk}: {kmer_fp}")

        # load core into dict by Seq_ID
        core_rows = {}
        for r in read_tsv_gz(core_fp):
            core_rows[r["Seq_ID"]] = r

        # prepare output rows
        out_rows = []
        for r in read_tsv_gz(kmer_fp):
            sid = r["Seq_ID"]
            core = core_rows.get(sid)
            if core is None:
                continue

            # init kmer columns with 0
            kmvals = {c: 0 for c in all_kmer_cols}

            for scope, col in [("kmer_all","kmer_all_json"), ("kmer_head","kmer_head_json"), ("kmer_tail","kmer_tail_json")]:
                js = r.get(col, "") or "{}"
                try:
                    obj = json.loads(js)
                except Exception:
                    obj = {}
                for k_str, cmap in obj.items():
                    k = int(k_str)
                    if not isinstance(cmap, dict):
                        continue
                    for kmer, cnt in cmap.items():
                        key = (scope, k, kmer)
                        colname = vocab.get(key)
                        if colname is not None:
                            kmvals[colname] += int(cnt)

            merged = dict(core)  # core already has Seq_ID and features
            merged.update(kmvals)
            out_rows.append(merged)

        # field order: Seq_ID + sorted core fields (excluding Seq_ID) + kmers
        core_fields = list(core_rows[next(iter(core_rows))].keys())
        core_fields = ["Seq_ID"] + sorted([c for c in core_fields if c != "Seq_ID"])
        fieldnames = core_fields + all_kmer_cols

        out_fp = out_dir / f"{chunk}.final.tsv.gz"
        write_tsv_gz(out_fp, fieldnames, out_rows)
        print(f"[DONE] {chunk}: rows={len(out_rows)} -> {out_fp}")

if __name__ == "__main__":
    main()
