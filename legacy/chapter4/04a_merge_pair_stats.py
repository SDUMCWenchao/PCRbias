#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

def open_tsv(fp: Path):
    if str(fp).endswith(".gz"):
        return gzip.open(fp, "rt", encoding="utf-8", newline="")
    return fp.open("r", encoding="utf-8", newline="")

def write_tsv_gz(fp: Path, header: List[str], rows: List[Dict[str, str]]):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fp, "wt", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=header, delimiter="\t")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "NA") for k in header}
            w.writerow(out)

def norm_key(row: Dict[str, str]) -> Optional[Tuple[str, str]]:
    pid = (row.get("pair_id") or "").strip()
    feat = (row.get("feature") or "").strip()
    if not pid or not feat:
        return None
    return pid, feat

def read_table(fp: Path,
               include: Optional[re.Pattern],
               exclude: Optional[re.Pattern]) -> Tuple[List[str], Dict[Tuple[str, str], Dict[str, str]]]:
    with open_tsv(fp) as f:
        r = csv.DictReader(f, delimiter="\t")
        if r.fieldnames is None:
            raise ValueError(f"Empty header: {fp}")
        header = list(r.fieldnames)

        data: Dict[Tuple[str, str], Dict[str, str]] = {}
        for row in r:
            k = norm_key(row)
            if k is None:
                continue
            feat = k[1]
            if include and not include.search(feat):
                continue
            if exclude and exclude.search(feat):
                continue
            data[k] = row
    return header, data

def merge_tables(a: Dict[Tuple[str, str], Dict[str, str]],
                 b: Dict[Tuple[str, str], Dict[str, str]],
                 prefer: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    out = dict(a)
    for k, row in b.items():
        if k not in out:
            out[k] = row
        else:
            if prefer == "second":
                out[k] = row
    return out

def union_header(h1: List[str], h2: List[str]) -> List[str]:
    seen = set()
    out = []
    for h in h1 + h2:
        if h not in seen:
            seen.add(h)
            out.append(h)
    # ensure pair_id, feature in front
    for must in ("pair_id", "feature"):
        if must in out:
            out.remove(must)
    return ["pair_id", "feature"] + out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", required=True, help="First input tsv(.gz)")
    ap.add_argument("--second", required=True, help="Second input tsv(.gz)")
    ap.add_argument("--out", required=True, help="Output .tsv.gz")
    ap.add_argument("--prefer", choices=["first", "second"], default="second",
                    help="When duplicated (pair_id,feature) exists, keep which one")
    ap.add_argument("--include_regex", default=None, help="Only keep features matching regex")
    ap.add_argument("--exclude_regex", default=None, help="Drop features matching regex")
    args = ap.parse_args()

    f1 = Path(args.first)
    f2 = Path(args.second)
    out = Path(args.out)

    if not f1.exists():
        raise FileNotFoundError(f"Missing: {f1}")
    if not f2.exists():
        raise FileNotFoundError(f"Missing: {f2}")

    inc = re.compile(args.include_regex) if args.include_regex else None
    exc = re.compile(args.exclude_regex) if args.exclude_regex else None

    h1, d1 = read_table(f1, inc, exc)
    h2, d2 = read_table(f2, inc, exc)

    merged = merge_tables(d1, d2, args.prefer)
    header = union_header(h1, h2)

    # stable ordering
    rows = [merged[k] for k in sorted(merged.keys(), key=lambda x: (x[0], x[1]))]
    write_tsv_gz(out, header, rows)

    print(f"[DONE] merged rows={len(rows)} -> {out}")

if __name__ == "__main__":
    main()
