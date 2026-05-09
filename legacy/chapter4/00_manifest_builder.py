#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 0: Build a manifest.tsv from samples_meta.tsv by locating each sample's
single-end FASTQ (.fq/.fastq/.gz) in raw_data_trimmed (preferred) or raw_data.

Outputs:
  <out_dir>/manifest.tsv

Manifest columns:
  file_id, sample_name, species, n_individuals, locus, pcr,
  input_type, fastq_path, source_dir, match_rule, n_matches, all_matches, status
"""

from __future__ import annotations
import argparse
import csv
import gzip
import os
from pathlib import Path
from typing import List, Tuple, Optional

FASTQ_EXTS_DEFAULT = [".fq", ".fastq", ".fq.gz", ".fastq.gz"]

def read_tsv_rows(tsv_path: Path) -> List[dict]:
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = [r for r in reader]
    required = {"file_id", "sample_name", "species", "n_individuals", "locus", "pcr"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise ValueError(f"Meta TSV missing required columns: {sorted(missing)}")
    return rows

def normpath(p: Path) -> str:
    return str(p.resolve())

def file_size_bytes(p: Path) -> int:
    try:
        return p.stat().st_size
    except Exception:
        return -1

def rank_candidate(path: Path, file_id: str, sample_name: str, base_dir: Path) -> int:
    """
    Higher rank = better.
    Prefer:
      1) exact name == file_id + ext at base_dir root
      2) exact name == sample_name + ext at base_dir root
      3) name startswith file_id
      4) name contains file_id
      5) name startswith sample_name
      6) name contains sample_name
    Also prefer shallower paths (closer to base_dir).
    """
    name = path.name
    rel_parts = path.relative_to(base_dir).parts if base_dir in path.parents or path == base_dir else ()
    depth_penalty = len(rel_parts)  # lower is better

    ext_ok = any(name.endswith(ext) for ext in FASTQ_EXTS_DEFAULT)  # loose check
    if not ext_ok:
        return -10_000

    # exact matches
    for ext in FASTQ_EXTS_DEFAULT:
        if name == f"{file_id}{ext}" and depth_penalty == 1:
            return 10_000
        if name == f"{sample_name}{ext}" and depth_penalty == 1:
            return 9_500

    # startswith/contains
    score = 0
    if name.startswith(file_id):
        score += 8_000
    elif file_id in name:
        score += 7_000

    if name.startswith(sample_name):
        score += 6_000
    elif sample_name in name:
        score += 5_000

    # common single-end conventions
    for tag in ["_R1", "_1", ".R1", ".1"]:
        for ext in FASTQ_EXTS_DEFAULT:
            if name == f"{file_id}{tag}{ext}":
                score += 7_500
            if name == f"{sample_name}{tag}{ext}":
                score += 6_500

    # shallower is better
    score -= depth_penalty * 10
    # slightly prefer larger file (avoid empty)
    sz = file_size_bytes(path)
    if sz >= 0:
        score += min(sz, 10_000_000) // 1_000_000  # +0..+10 approx

    return score

def collect_candidates(base_dir: Path, file_id: str, sample_name: str, exts: List[str]) -> List[Path]:
    """
    Collect candidate files in base_dir (including subdirs) using conservative patterns.
    """
    candidates: List[Path] = []

    # 1) direct exact checks (fast)
    for ext in exts:
        for stem in [file_id, sample_name, f"{file_id}_R1", f"{file_id}_1", f"{sample_name}_R1", f"{sample_name}_1"]:
            p = base_dir / f"{stem}{ext}"
            if p.exists() and p.is_file():
                candidates.append(p)

    # 2) non-recursive glob (usually enough)
    patterns = []
    for ext in exts:
        patterns.extend([
            f"{file_id}*{ext}",
            f"{sample_name}*{ext}",
        ])
    for pat in patterns:
        for p in base_dir.glob(pat):
            if p.exists() and p.is_file():
                candidates.append(p)

    # 3) recursive search (fallback; can be slower)
    # Limit patterns to avoid explosion
    recursive_patterns = []
    for ext in exts:
        recursive_patterns.extend([
            f"{file_id}*{ext}",
            f"{sample_name}*{ext}",
        ])
    for pat in recursive_patterns:
        # rglob includes subdirs
        for p in base_dir.rglob(pat):
            if p.exists() and p.is_file():
                candidates.append(p)

    # de-duplicate
    uniq = {}
    for p in candidates:
        uniq[normpath(p)] = p
    return list(uniq.values())

def choose_best(cands: List[Path], file_id: str, sample_name: str, base_dir: Path) -> Tuple[Optional[Path], List[Path]]:
    if not cands:
        return None, []
    scored = [(rank_candidate(p, file_id, sample_name, base_dir), p) for p in cands]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    # Keep "top tier" as ambiguous if multiple share same best score
    top = [p for s, p in scored if s == best_score]
    return best_path, scored_to_paths(scored)

def scored_to_paths(scored: List[Tuple[int, Path]]) -> List[Path]:
    return [p for _, p in scored]

def infer_match_rule(best: Path, file_id: str, sample_name: str) -> str:
    name = best.name
    for ext in FASTQ_EXTS_DEFAULT:
        if name == f"{file_id}{ext}":
            return "exact_file_id"
        if name == f"{sample_name}{ext}":
            return "exact_sample_name"
        for tag in ["_R1", "_1", ".R1", ".1"]:
            if name == f"{file_id}{tag}{ext}":
                return "exact_file_id_tag"
            if name == f"{sample_name}{tag}{ext}":
                return "exact_sample_name_tag"
    if name.startswith(file_id):
        return "glob_startswith_file_id"
    if file_id in name:
        return "glob_contains_file_id"
    if name.startswith(sample_name):
        return "glob_startswith_sample_name"
    if sample_name in name:
        return "glob_contains_sample_name"
    return "unknown"

def main():
    ap = argparse.ArgumentParser(description="Step 0: build manifest.tsv by locating single-end FASTQ files.")
    ap.add_argument("--project_dir", default=None, help="Project root. Default: parent of this script directory.")
    ap.add_argument("--meta", default=None, help="Path to samples_meta.tsv. Default: <project_dir>/samples_meta.tsv")
    ap.add_argument("--raw_dir", default=None, help="Raw data dir. Default: <project_dir>/raw_data")
    ap.add_argument("--trimmed_dir", default=None, help="Trimmed data dir. Default: <project_dir>/raw_data_trimmed")
    ap.add_argument("--out_dir", default=None, help="Output dir. Default: <project_dir>/analysis_results/00_manifest")
    ap.add_argument("--exts", default=",".join(FASTQ_EXTS_DEFAULT), help="Comma-separated FASTQ extensions to search.")
    ap.add_argument("--prefer_trimmed", action="store_true", help="Prefer trimmed dir (default).")
    ap.add_argument("--prefer_raw", action="store_true", help="Prefer raw dir instead.")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).resolve() if args.project_dir else script_dir.parent

    meta = Path(args.meta).resolve() if args.meta else (project_dir / "samples_meta.tsv")
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else (project_dir / "raw_data")
    trimmed_dir = Path(args.trimmed_dir).resolve() if args.trimmed_dir else (project_dir / "raw_data_trimmed")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_dir / "analysis_results" / "00_manifest")

    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    if not exts:
        raise ValueError("No extensions provided via --exts")

    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta.exists():
        raise FileNotFoundError(f"Meta TSV not found: {meta}")
    if not raw_dir.exists():
        print(f"[WARN] raw_dir does not exist: {raw_dir}")
    if not trimmed_dir.exists():
        print(f"[WARN] trimmed_dir does not exist: {trimmed_dir}")

    prefer_trimmed = True
    if args.prefer_raw:
        prefer_trimmed = False
    if args.prefer_trimmed:
        prefer_trimmed = True

    search_dirs = [trimmed_dir, raw_dir] if prefer_trimmed else [raw_dir, trimmed_dir]
    search_dirs = [d for d in search_dirs if d.exists()]

    rows = read_tsv_rows(meta)
    manifest_path = out_dir / "manifest.tsv"

    found = 0
    missing = 0
    ambiguous = 0

    fieldnames = [
        "file_id", "sample_name", "species", "n_individuals", "locus", "pcr",
        "input_type", "fastq_path", "source_dir", "match_rule",
        "n_matches", "all_matches", "status"
    ]

    with manifest_path.open("w", encoding="utf-8", newline="") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()

        for r in rows:
            file_id = (r.get("file_id") or "").strip()
            sample_name = (r.get("sample_name") or "").strip()
            if not file_id:
                raise ValueError("Encountered empty file_id in meta TSV")

            best_path = None
            best_dir = None
            all_ranked: List[Path] = []
            all_matches: List[Path] = []

            for d in search_dirs:
                cands = collect_candidates(d, file_id, sample_name, exts)
                if cands:
                    best, ranked = choose_best(cands, file_id, sample_name, d)
                    if best is not None:
                        best_path = best
                        best_dir = d
                        all_ranked = ranked
                        all_matches = cands
                        break

            status = "OK"
            match_rule = ""
            fastq_path = ""
            source_dir = ""
            n_matches = 0
            all_matches_str = ""

            if best_path is None:
                status = "MISSING"
                missing += 1
            else:
                found += 1
                fastq_path = normpath(best_path)
                source_dir = "trimmed" if best_dir and best_dir == trimmed_dir else "raw"
                match_rule = infer_match_rule(best_path, file_id, sample_name)
                n_matches = len(all_matches)
                # store top 20 ranked matches
                top_ranked = all_ranked[:20]
                all_matches_str = ";".join(normpath(p) for p in top_ranked)

                if n_matches > 1:
                    status = "AMBIGUOUS"
                    ambiguous += 1

            out_row = {
                "file_id": file_id,
                "sample_name": r.get("sample_name", ""),
                "species": r.get("species", ""),
                "n_individuals": r.get("n_individuals", ""),
                "locus": r.get("locus", ""),
                "pcr": r.get("pcr", ""),
                "input_type": "SE_FASTQ",
                "fastq_path": fastq_path,
                "source_dir": source_dir,
                "match_rule": match_rule,
                "n_matches": str(n_matches),
                "all_matches": all_matches_str,
                "status": status,
            }
            writer.writerow(out_row)

    print(f"[DONE] Wrote: {manifest_path}")
    print(f"[SUMMARY] total={len(rows)} found={found} missing={missing} ambiguous={ambiguous}")
    if missing > 0:
        print("         -> Please check missing samples in manifest.tsv (status=MISSING)")
    if ambiguous > 0:
        print("         -> Please check ambiguous samples in manifest.tsv (status=AMBIGUOUS)")

if __name__ == "__main__":
    main()
