#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import pandas as pd

CANONICAL_LABELS = {
    "Bos_taurus",
    "Sus_scrofa",
    "Equus_asinus",
    "Canis_lupus",
    "Felis_catus",
    "Homo_sapiens",
    "Mus_musculus",
    "Oryctolagus_cuniculus",
    "Ovis_aries",
    "Capra_hircus",
}

ALIAS_TO_CANONICAL = {
    "Bos taurus": "Bos_taurus",
    "Bos_taurus_taurus": "Bos_taurus",
    "Sus scrofa": "Sus_scrofa",
    "Sus_scrofa_domesticus": "Sus_scrofa",
    "Equus asinus": "Equus_asinus",
    "Canis lupus": "Canis_lupus",
    "Canis_lupus_familiaris": "Canis_lupus",
    "Canis familiaris": "Canis_lupus",
    "Felis catus": "Felis_catus",
    "Homo sapiens": "Homo_sapiens",
    "Mus musculus": "Mus_musculus",
    "Mus_musculus_domesticus": "Mus_musculus",
    "Oryctolagus cuniculus": "Oryctolagus_cuniculus",
    "Ovis aries": "Ovis_aries",
    "Capra hircus": "Capra_hircus",
}

TARGET_FILES = [
    ("03_tables/annotation_online/sequence_annotation_template.tsv", ["species_label"]),
    ("03_tables/chapter2_design/expected_inter_targets_template.tsv", ["target_label"]),
    ("03_tables/chapter2_design/expected_intra_targets.tsv", ["target_label"]),
    ("03_tables/annotation/master_long_abundance_annotated.tsv", ["species_label"]),
    ("05_stats/chapter2/bias_and_nontarget/inter_species_bias.tsv", ["species_label"]),
    ("03_tables/annotation_online/online_blast_annotation_template.tsv", ["species_label"]),
]

def normalize_basic(text: str) -> str:
    s = str(text).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return s
    s = re.sub(r"[ \t/|:-]+", "_", s)
    s = re.sub(r"__+", "_", s)
    s = s.strip("_")
    s = re.sub(r"\(.*?\)", "", s).strip("_ ")
    parts = [p for p in s.split("_") if p]
    if not parts:
        return ""
    parts[0] = parts[0][0].upper() + parts[0][1:].lower() if parts[0] else parts[0]
    for i in range(1, len(parts)):
        parts[i] = parts[i].lower()
    s = "_".join(parts)
    s = re.sub(r"__+", "_", s).strip("_")
    return s

def unify_label(raw: str):
    raw = "" if pd.isna(raw) else str(raw)
    basic = normalize_basic(raw)
    if basic in ALIAS_TO_CANONICAL:
        return ALIAS_TO_CANONICAL[basic], "alias_map"
    raw_compact = re.sub(r"[\s_]+", "", raw).lower()
    for k, v in ALIAS_TO_CANONICAL.items():
        if re.sub(r"[\s_]+", "", k).lower() == raw_compact:
            return v, "alias_map_compact"
    if basic.startswith("Canis_lupus_"):
        return "Canis_lupus", "trinomial_to_binomial"
    if basic.startswith("Mus_musculus_"):
        return "Mus_musculus", "trinomial_to_binomial"
    if basic.startswith("Sus_scrofa_"):
        return "Sus_scrofa", "trinomial_to_binomial"
    if basic.startswith("Bos_taurus_"):
        return "Bos_taurus", "trinomial_to_binomial"
    return basic, "basic_normalize"

def process_one_file(path: Path, columns, dry_run=False, backup=False):
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    changes = []
    unmatched = []
    for col in columns:
        if col not in df.columns:
            continue
        new_vals = []
        for idx, val in enumerate(df[col].tolist()):
            new_val, rule = unify_label(val)
            new_vals.append(new_val)
            if str(val) != str(new_val):
                changes.append({
                    "file": str(path),
                    "row_number_1based": idx + 2,
                    "column": col,
                    "old_value": val,
                    "new_value": new_val,
                    "rule": rule,
                })
            if new_val and new_val not in CANONICAL_LABELS and col in {"species_label", "target_label"}:
                unmatched.append({
                    "file": str(path),
                    "row_number_1based": idx + 2,
                    "column": col,
                    "value_after_normalization": new_val,
                    "note": "not in current canonical label set; check whether expected",
                })
        df[col] = new_vals
    if not dry_run:
        if backup:
            bak = path.with_suffix(path.suffix + ".bak")
            if not bak.exists():
                import shutil
                shutil.copy2(path, bak)
        df.to_csv(path, sep="\t", index=False)
    return pd.DataFrame(changes), pd.DataFrame(unmatched)

def main():
    ap = argparse.ArgumentParser(description="Check and unify species/target labels across Chapter 2 files, overwrite originals in place.")
    ap.add_argument("--base-dir", required=True, help="e.g. /path/to/chapter2_3_analysis")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backup", action="store_true")
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    report_dir = base_dir / "10_docs" / "label_unify_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_changes = []
    all_unmatched = []
    processed = []

    for rel_path, cols in TARGET_FILES:
        path = base_dir / rel_path
        if not path.exists():
            continue
        chg, um = process_one_file(path, cols, dry_run=args.dry_run, backup=args.backup)
        all_changes.append(chg)
        all_unmatched.append(um)
        processed.append({
            "file": str(path),
            "columns_checked": ",".join(cols),
            "n_changes": len(chg),
            "n_unmatched": len(um),
        })

    processed_df = pd.DataFrame(processed)
    if all_changes:
        changes_df = pd.concat(all_changes, ignore_index=True)
    else:
        changes_df = pd.DataFrame(columns=["file","row_number_1based","column","old_value","new_value","rule"])
    if all_unmatched:
        unmatched_df = pd.concat(all_unmatched, ignore_index=True)
    else:
        unmatched_df = pd.DataFrame(columns=["file","row_number_1based","column","value_after_normalization","note"])

    processed_df.to_csv(report_dir / "label_unify_file_summary.tsv", sep="\t", index=False)
    changes_df.to_csv(report_dir / "label_unify_changes.tsv", sep="\t", index=False)
    unmatched_df.to_csv(report_dir / "label_unify_unmatched.tsv", sep="\t", index=False)

    print(f"Wrote {report_dir / 'label_unify_file_summary.tsv'}")
    print(f"Wrote {report_dir / 'label_unify_changes.tsv'}")
    print(f"Wrote {report_dir / 'label_unify_unmatched.tsv'}")
    print()
    if len(processed_df):
        print(processed_df.to_string(index=False))
    else:
        print("No target files found.")
    print()
    print("Mode:", "DRY RUN" if args.dry_run else "OVERWRITE IN PLACE")
    print("Backup:", "ON" if args.backup else "OFF")

if __name__ == "__main__":
    main()
