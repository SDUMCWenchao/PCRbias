#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 0b: Validate manifest.tsv

Checks:
- status != MISSING
- fastq_path exists and readable
- duplicates (same fastq_path used by multiple file_ids)
- quick FASTQ format sanity check (first record): @header, + line, seq/qual length match

Outputs:
  <out_dir>/validation_report.txt
Exit code:
  0 if OK (no missing, no format errors)
  2 if has missing or format errors
"""

from __future__ import annotations
import argparse
import csv
import gzip
from pathlib import Path
from typing import Dict, List, Tuple

def open_text_maybe_gz(p: Path):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return p.open("r", encoding="utf-8", errors="replace")

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

def fastq_sanity_check(fastq_path: Path) -> Tuple[bool, str]:
    """
    Read first 4 lines (one record). Basic checks only.
    """
    try:
        with open_text_maybe_gz(fastq_path) as f:
            l1 = f.readline().rstrip("\n")
            l2 = f.readline().rstrip("\n")
            l3 = f.readline().rstrip("\n")
            l4 = f.readline().rstrip("\n")
    except Exception as e:
        return False, f"cannot_open: {e}"

    if not (l1 and l2 and l3 and l4):
        return False, "file_too_short_for_fastq_record"
    if not l1.startswith("@"):
        return False, "line1_not_header(@)"
    if not l3.startswith("+"):
        return False, "line3_not_plus(+)"
    if len(l2) != len(l4):
        return False, f"seq_qual_length_mismatch(seq={len(l2)} qual={len(l4)})"
    return True, "ok"

def main():
    ap = argparse.ArgumentParser(description="Step 0b: validate manifest.tsv")
    ap.add_argument("--manifest", default=None, help="Path to manifest.tsv. Default: <project_dir>/analysis_results/00_manifest/manifest.tsv")
    ap.add_argument("--project_dir", default=None, help="Project root. Default: parent of this script directory.")
    ap.add_argument("--out_dir", default=None, help="Output dir for validation_report.txt. Default: same dir as manifest.")
    ap.add_argument("--skip_fastq_check", action="store_true", help="Skip FASTQ sanity check (format).")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).resolve() if args.project_dir else script_dir.parent

    manifest_path = Path(args.manifest).resolve() if args.manifest else (project_dir / "analysis_results" / "00_manifest" / "manifest.tsv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "validation_report.txt"

    rows = read_manifest(manifest_path)

    missing_rows = []
    not_found_paths = []
    duplicates: Dict[str, List[str]] = {}
    format_errors = []

    # duplicates
    path_to_ids: Dict[str, List[str]] = {}
    for r in rows:
        fid = (r.get("file_id") or "").strip()
        p = (r.get("fastq_path") or "").strip()
        if p:
            path_to_ids.setdefault(p, []).append(fid)

    for p, ids in path_to_ids.items():
        if len(ids) > 1:
            duplicates[p] = ids

    # existence + format checks
    for r in rows:
        fid = (r.get("file_id") or "").strip()
        status = (r.get("status") or "").strip()
        p_str = (r.get("fastq_path") or "").strip()

        if status == "MISSING" or not p_str:
            missing_rows.append(fid)
            continue

        p = Path(p_str)
        if not p.exists():
            not_found_paths.append((fid, p_str))
            continue

        if not args.skip_fastq_check:
            ok, msg = fastq_sanity_check(p)
            if not ok:
                format_errors.append((fid, p_str, msg))

    # write report
    total = len(rows)
    n_missing = len(missing_rows)
    n_not_found = len(not_found_paths)
    n_dup = len(duplicates)
    n_fmt = len(format_errors)

    with report_path.open("w", encoding="utf-8") as fo:
        fo.write(f"Manifest validation report\n")
        fo.write(f"manifest: {manifest_path}\n\n")
        fo.write(f"SUMMARY\n")
        fo.write(f"  total_rows: {total}\n")
        fo.write(f"  missing_status_rows: {n_missing}\n")
        fo.write(f"  path_not_found_rows: {n_not_found}\n")
        fo.write(f"  duplicate_paths: {n_dup}\n")
        fo.write(f"  fastq_format_errors: {n_fmt}\n\n")

        if n_missing:
            fo.write("MISSING (status=MISSING or empty path)\n")
            for fid in missing_rows:
                fo.write(f"  - {fid}\n")
            fo.write("\n")

        if n_not_found:
            fo.write("PATH NOT FOUND (manifest points to non-existent file)\n")
            for fid, p in not_found_paths:
                fo.write(f"  - {fid}\t{p}\n")
            fo.write("\n")

        if n_dup:
            fo.write("DUPLICATE FASTQ PATHS (same file used by multiple file_id rows)\n")
            for p, ids in duplicates.items():
                fo.write(f"  - {p}\tfile_ids={','.join(ids)}\n")
            fo.write("\n")

        if n_fmt:
            fo.write("FASTQ FORMAT ERRORS (first record sanity check)\n")
            for fid, p, msg in format_errors:
                fo.write(f"  - {fid}\t{p}\t{msg}\n")
            fo.write("\n")

    print(f"[DONE] Wrote: {report_path}")
    print(f"[SUMMARY] total={total} missing={n_missing} not_found={n_not_found} dup_paths={n_dup} fastq_fmt_errors={n_fmt}")

    # exit code
    if n_missing or n_not_found or n_fmt:
        raise SystemExit(2)
    raise SystemExit(0)

if __name__ == "__main__":
    main()
