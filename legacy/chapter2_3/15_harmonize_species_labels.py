#!/usr/bin/env python3
from pathlib import Path
import argparse, re
import pandas as pd

DEFAULT_ALIASES = {
    "Canis_lupus_familiaris": "Canis_lupus",
    "Canis familiaris": "Canis_lupus",
    "Canis_familiaris": "Canis_lupus",
    "dog": "Canis_lupus",
    "Dog": "Canis_lupus",
    "Mus_musculus_domesticus": "Mus_musculus",
    "Mus musculus domesticus": "Mus_musculus",
    "Oryctolagus cuniculus": "Oryctolagus_cuniculus",
    "Homo sapiens": "Homo_sapiens",
    "Bos taurus": "Bos_taurus",
    "Sus scrofa": "Sus_scrofa",
    "Equus asinus": "Equus_asinus",
    "Ovis aries": "Ovis_aries",
    "Capra hircus": "Capra_hircus",
    "Felis catus": "Felis_catus",
}

def normalize_basic(x: str) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"[;|]+$", "", s)
    s = s.replace("-", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def load_alias_table(path):
    aliases = dict(DEFAULT_ALIASES)
    if path:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p, sep="\t", dtype=str).fillna("")
            if not {"alias","canonical"}.issubset(df.columns):
                raise SystemExit("Alias table must contain columns: alias, canonical")
            for _, r in df.iterrows():
                a = str(r["alias"]).strip()
                c = str(r["canonical"]).strip()
                if a:
                    aliases[a] = c
    return aliases

def harmonize_value(v, aliases):
    if pd.isna(v):
        return v
    raw = str(v).strip()
    if raw == "":
        return raw
    step1 = normalize_basic(raw)
    if raw in aliases:
        return aliases[raw]
    if step1 in aliases:
        return aliases[step1]
    return step1

def maybe_harmonize_column(df, col, aliases):
    if col not in df.columns:
        return df, pd.DataFrame(columns=["before","after"])
    out = df.copy()
    before = out[col].astype(str).fillna("")
    after = before.map(lambda x: harmonize_value(x, aliases))
    changes = pd.DataFrame({"before": before, "after": after})
    changes = changes[changes["before"] != changes["after"]].copy()
    out[col] = after
    return out, changes

def collect_unique(df, cols, source_name):
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        vals = sorted({str(x).strip() for x in df[c].dropna().astype(str) if str(x).strip() != ""})
        for v in vals:
            rows.append({"source_file": source_name, "column": c, "label": v})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Check and harmonize species labels across PCR-bias Chapter 2 files.")
    ap.add_argument("--annotation-online", required=True)
    ap.add_argument("--expected-inter", required=True)
    ap.add_argument("--expected-intra", required=True)
    ap.add_argument("--annotated-abundance", default="")
    ap.add_argument("--alias-table", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    aliases = load_alias_table(args.alias_table if args.alias_table else None)

    file_map = {
        "annotation_online": (Path(args.annotation_online), ["species_label"]),
        "expected_inter": (Path(args.expected_inter), ["target_label"]),
        "expected_intra": (Path(args.expected_intra), ["target_label"]),
    }
    if args.annotated_abundance:
        file_map["annotated_abundance"] = (Path(args.annotated_abundance), ["species_label"])

    loaded = {}
    before_parts = []
    for key, (path, cols) in file_map.items():
        df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
        loaded[key] = (path, cols, df)
        before_parts.append(collect_unique(df, cols, path.name))
    before_df = pd.concat(before_parts, ignore_index=True) if before_parts else pd.DataFrame(columns=["source_file","column","label"])
    before_df.to_csv(outdir / "label_unique_values_before.tsv", sep="\t", index=False)

    corrected = {}
    change_parts = []
    after_parts = []
    for key, (path, cols, df) in loaded.items():
        cur = df.copy()
        for c in cols:
            cur, changes = maybe_harmonize_column(cur, c, aliases)
            if not changes.empty:
                changes = changes.copy()
                changes["source_file"] = path.name
                changes["column"] = c
                change_parts.append(changes)
        corrected[key] = (path, cur)
        after_parts.append(collect_unique(cur, cols, path.name))

    after_df = pd.concat(after_parts, ignore_index=True) if after_parts else pd.DataFrame(columns=["source_file","column","label"])
    after_df.to_csv(outdir / "label_unique_values_after.tsv", sep="\t", index=False)

    changes_df = pd.concat(change_parts, ignore_index=True) if change_parts else pd.DataFrame(columns=["before","after","source_file","column"])
    if not changes_df.empty:
        changes_df = changes_df[["source_file","column","before","after"]].drop_duplicates().sort_values(["source_file","column","before","after"])
    changes_df.to_csv(outdir / "label_harmonization_changes.tsv", sep="\t", index=False)

    ann_labels = set()
    if "annotation_online" in corrected:
        ann_df = corrected["annotation_online"][1]
        ann_labels = {str(x).strip() for x in ann_df["species_label"].astype(str).tolist() if str(x).strip()}
    coverage_rows = []
    for file_key in ["expected_inter", "expected_intra"]:
        if file_key in corrected:
            path, df = corrected[file_key]
            for _, r in df.iterrows():
                tgt = str(r.get("target_label","")).strip()
                coverage_rows.append({
                    "source_file": path.name,
                    "sample_id": r.get("sample_id", ""),
                    "marker": r.get("marker", ""),
                    "group_name": r.get("group_name", ""),
                    "target_label": tgt,
                    "present_in_annotation_online": tgt in ann_labels if ann_labels else "",
                })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(outdir / "expected_label_coverage_after_harmonization.tsv", sep="\t", index=False)
    missing_df = coverage_df[coverage_df["present_in_annotation_online"] == False].copy() if not coverage_df.empty else pd.DataFrame(columns=["source_file","sample_id","marker","group_name","target_label","present_in_annotation_online"])
    missing_df.to_csv(outdir / "expected_label_missing_after_harmonization.tsv", sep="\t", index=False)

    pd.DataFrame([{"alias":k, "canonical":v} for k,v in sorted(aliases.items())]).to_csv(outdir / "aliases_used.tsv", sep="\t", index=False)

    if args.apply:
        applied_dir = outdir / "applied_corrected_files"
        applied_dir.mkdir(parents=True, exist_ok=True)
        for key, (orig_path, dfcorr) in corrected.items():
            dfcorr.to_csv(applied_dir / orig_path.name, sep="\t", index=False)

    summary = pd.DataFrame([{
        "n_unique_before": int(before_df["label"].nunique()) if not before_df.empty else 0,
        "n_unique_after": int(after_df["label"].nunique()) if not after_df.empty else 0,
        "n_changes": int(len(changes_df)),
        "apply_mode_used": bool(args.apply),
    }])
    summary.to_csv(outdir / "harmonization_summary.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'harmonization_summary.tsv'}")
    print(f"Wrote {outdir / 'label_harmonization_changes.tsv'}")
    print(f"Wrote {outdir / 'expected_label_missing_after_harmonization.tsv'}")
    if args.apply:
        print(f"Wrote corrected files under {outdir / 'applied_corrected_files'}")

if __name__ == "__main__":
    main()
