#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, gzip, re
from pathlib import Path
from collections import Counter

IUPAC_OK = set(list("ACGTNRYKMSWBDHV"))
HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")

def clean_token(s: str) -> str:
    x = (s or "").strip()
    for _ in range(2):
        if len(x) >= 2 and ((x[0] == '"' and x[-1] == '"') or (x[0] == "'" and x[-1] == "'")):
            x = x[1:-1].strip()
    if x.startswith(">"):
        x = x[1:].strip()
    return x

def looks_like_seq(s: str) -> bool:
    if len(s) < 15:
        return False
    ss = clean_token(s).upper().replace("U","T").replace(" ","")
    return all(c in IUPAC_OK for c in ss)

def sniff_abundance_format(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip("\n").split(",")
    lower = [h.strip().lower() for h in header]
    has_exp = any(x in lower for x in ["experiment","exp","sample","condition"])
    has_seq = any(x in lower for x in ["sequence","seq","seq_id","scaffold","id"])
    has_val = any(x in lower for x in ["abundance","count","reads","value"])
    if has_exp and has_seq and has_val:
        return "long"
    return "wide"

def sample_ids_from_abundance(csv_path: Path, n=2000):
    fmt = sniff_abundance_format(csv_path)
    ids = []
    if fmt == "wide":
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rr = csv.reader(f)
            header = next(rr, None)
            if not header:
                return fmt, [], []
            exps = [h.strip() for h in header[1:]]
            for row in rr:
                if not row: 
                    continue
                ids.append(clean_token(row[0]))
                if len(ids) >= n:
                    break
            return fmt, exps, ids
    # long
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f)
        cols = {c.lower(): c for c in (dr.fieldnames or [])}
        def pick(cands):
            for c in cands:
                if c in cols: return cols[c]
            return None
        col_seq = pick(["sequence","seq","seq_id","scaffold","id"])
        col_exp = pick(["experiment","exp","sample","condition"])
        exps = set()
        for row in dr:
            if col_exp:
                exps.add((row.get(col_exp) or "").strip())
            if col_seq:
                ids.append(clean_token(row.get(col_seq) or ""))
            if len(ids) >= n:
                break
    return fmt, sorted([e for e in exps if e]), ids

def list_candidate_mapping_files(ds_dir: Path, abundance_name: str):
    cands = []
    # fasta-ish
    for pat in ["*.fa", "*.fna", "*.fasta", "*.fa.gz", "*.fna.gz", "*.fasta.gz"]:
        cands += list(ds_dir.glob(pat))
    # tsv/csv-ish
    for pat in ["*.tsv", "*.tsv.gz", "*.csv", "*.csv.gz"]:
        cands += list(ds_dir.glob(pat))
    # filter: remove abundance_by_experiment itself
    cands = [p for p in cands if p.name != abundance_name]
    # prefer names containing seq/sequence/oligo/map
    def score(p: Path):
        nm = p.name.lower()
        s = 0
        for k in ["seq", "sequence", "oligo", "map", "id", "uniq", "fasta", "fa"]:
            if k in nm: s += 2
        if "abundance" in nm: s -= 5
        return s
    cands = sorted(set(cands), key=score, reverse=True)
    return cands[:30]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcrbias_root", required=True)
    ap.add_argument("--dataset_glob", default="analysis/data/external_datasets/*")
    ap.add_argument("--abundance_name", default="abundance_by_experiment.csv")
    ap.add_argument("--sample_n", type=int, default=2000)
    args = ap.parse_args()

    root = Path(args.pcrbias_root).resolve()
    ds_dirs = sorted([d for d in root.glob(args.dataset_glob) if d.is_dir() and (d / args.abundance_name).exists()])
    print(f"[INFO] datasets={len(ds_dirs)}")
    for d in ds_dirs:
        ab = d / args.abundance_name
        fmt, exps, ids = sample_ids_from_abundance(ab, n=args.sample_n)
        cnt = Counter()
        for x in ids:
            if not x:
                cnt["empty"] += 1
            elif looks_like_seq(x):
                cnt["seq_like"] += 1
            elif HEX32.match(x):
                cnt["md5_like"] += 1
            elif x.isdigit():
                cnt["numeric"] += 1
            else:
                cnt["other"] += 1
        print(f"\n=== {d.name} ===")
        print(f"[INFO] abundance fmt={fmt} exps={len(exps)} sample_ids={len(ids)} class={dict(cnt)}")
        print(f"[INFO] first5_ids={ids[:5]}")
        cands = list_candidate_mapping_files(d, args.abundance_name)
        print("[INFO] candidate mapping files (top):")
        for p in cands[:12]:
            print("  -", p.name)

if __name__ == "__main__":
    main()
