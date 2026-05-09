#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd

# 你要求的强制统一规则
EXACT_RULES = {
    "Canis_lupus_familiaris": ("Canis_lupus", "exact_rule:dog_to_wolf"),
}

# 默认近缘物种合并规则（保守版，但足够覆盖你当前 annotation_master.tsv 中最主要的问题）
NEAR_RELATIVE_RULES = {
    # cattle-like
    "Bos_indicus": ("Bos_taurus", "near_relative_rule:bovid_to_cattle"),
    "Bos_javanicus": ("Bos_taurus", "near_relative_rule:bovid_to_cattle"),
    "Bison_bonasus": ("Bos_taurus", "near_relative_rule:bovid_to_cattle"),
    "Bubalus_bubalis": ("Bos_taurus", "near_relative_rule:bovid_to_cattle"),

    # canid-like
    "Vulpes_bengalensis": ("Canis_lupus", "near_relative_rule:canid_to_wolf"),
    "Vulpes_rueppellii": ("Canis_lupus", "near_relative_rule:canid_to_wolf"),

    # equid-like
    "Equus_quagga": ("Equus_asinus", "near_relative_rule:equid_to_donkey"),

    # sheep/goat side
    "Ovis_orientalis": ("Ovis_aries", "near_relative_rule:ovis_to_sheep"),
    "Ammotragus_lervia": ("Ovis_aries", "near_relative_rule:caprine_to_sheep"),
    "Budorcas_taxicolor_whitei": ("Capra_hircus", "near_relative_rule:caprine_to_goat"),
}

# 可选：激进模式，把更多反刍/近缘 ungulates 并到最近目标类
AGGRESSIVE_RULES = {
    "Alces_alces": ("Bos_taurus", "aggressive_rule:ungulate_to_cattle"),
    "Odocoileus_virginianus": ("Bos_taurus", "aggressive_rule:ungulate_to_cattle"),
    "Rangifer_tarandus": ("Bos_taurus", "aggressive_rule:ungulate_to_cattle"),
    "Philantomba_monticola": ("Capra_hircus", "aggressive_rule:small_bovid_to_goat"),
    "Diceros_bicornis_minor": ("Bos_taurus", "aggressive_rule:large_ungulate_to_cattle"),
}

def pick_label(label: str, expected_targets: set[str], aggressive: bool = False):
    if pd.isna(label) or str(label).strip() == "":
        return "UNANNOTATED", "empty_label"

    label = str(label).strip()

    # already standardized
    if label in expected_targets:
        return label, "already_expected_target"

    # exact rules first
    if label in EXACT_RULES:
        return EXACT_RULES[label]

    # near-relative rules
    if label in NEAR_RELATIVE_RULES:
        return NEAR_RELATIVE_RULES[label]

    # same-genus fallback for expected genera
    genus = label.split("_")[0] if "_" in label else label
    genus_to_target = {
        "Bos": "Bos_taurus",
        "Sus": "Sus_scrofa",
        "Capra": "Capra_hircus",
        "Ovis": "Ovis_aries",
        "Equus": "Equus_asinus",
        "Felis": "Felis_catus",
        "Canis": "Canis_lupus",
        "Mus": "Mus_musculus",
        "Oryctolagus": "Oryctolagus_cuniculus",
        "Homo": "Homo_sapiens",
    }
    if genus in genus_to_target:
        return genus_to_target[genus], f"genus_fallback:{genus}"

    # aggressive merge if requested
    if aggressive and label in AGGRESSIVE_RULES:
        return AGGRESSIVE_RULES[label]

    # otherwise keep as is
    return label, "kept_as_original"

def main():
    ap = argparse.ArgumentParser(description="Check and one-click unify species labels against expected target labels.")
    ap.add_argument("--annotation-master", required=True, help="annotation_master.tsv")
    ap.add_argument("--expected-inter", required=True, help="expected_inter_targets_template.tsv")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--aggressive", action="store_true", help="merge additional ungulate-like near relatives more aggressively")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ann = pd.read_csv(args.annotation_master, sep="\t", dtype=str).fillna("")
    exp = pd.read_csv(args.expected_inter, sep="\t", dtype=str).fillna("")

    if "species_label" not in ann.columns:
        raise SystemExit("annotation_master.tsv missing species_label column")
    if "target_label" not in exp.columns:
        raise SystemExit("expected_inter_targets_template.tsv missing target_label column")

    expected_targets = set(exp["target_label"].dropna().astype(str).str.strip().unique())

    ann["species_label_original"] = ann["species_label"]
    mapped = ann["species_label_original"].apply(lambda x: pick_label(x, expected_targets, aggressive=args.aggressive))
    ann["species_label"] = mapped.apply(lambda x: x[0])
    ann["label_unify_reason"] = mapped.apply(lambda x: x[1])

    # keep annotation_status informative
    if "annotation_status" in ann.columns:
        ann["annotation_status_original"] = ann["annotation_status"]
        ann.loc[
            ann["species_label_original"] != ann["species_label"],
            "annotation_status"
        ] = "standardized_label"

    # mapping report
    mapping_report = (
        ann.groupby(["species_label_original","species_label","label_unify_reason"], as_index=False)
        .agg(
            n_sequences=("sequence_id", "nunique"),
            total_count=("total_count", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()) if "total_count" in ann.columns else ("sequence_id", "size")
        )
        .sort_values(["n_sequences","total_count"], ascending=[False, False])
    )

    # target coverage summary after unification
    ann["is_expected_target_after_unify"] = ann["species_label"].isin(expected_targets).astype(int)
    target_summary = (
        ann.groupby(["marker","is_expected_target_after_unify"], as_index=False)
        .agg(
            n_sequences=("sequence_id","nunique"),
            total_count=("total_count", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()) if "total_count" in ann.columns else ("sequence_id", "size")
        )
    )
    target_summary["class"] = target_summary["is_expected_target_after_unify"].map({1:"expected_target", 0:"other_label"})

    # unmatched labels report
    unmatched = ann.loc[~ann["species_label"].isin(expected_targets), [
        c for c in ["sequence_id","marker","species_label_original","species_label","label_unify_reason","best_hit_species","best_hit_description","total_count","max_rel_abundance"] if c in ann.columns
    ]].copy()
    unmatched = unmatched.sort_values([c for c in ["total_count","max_rel_abundance"] if c in unmatched.columns], ascending=False)

    # write outputs
    ann.to_csv(outdir / "annotation_master_unified.tsv", sep="\t", index=False)
    mapping_report.to_csv(outdir / "label_unify_mapping_report.tsv", sep="\t", index=False)
    target_summary.to_csv(outdir / "label_unify_target_coverage.tsv", sep="\t", index=False)
    unmatched.to_csv(outdir / "label_unify_unmatched_labels.tsv", sep="\t", index=False)

    print(f"Wrote {outdir / 'annotation_master_unified.tsv'}")
    print(f"Wrote {outdir / 'label_unify_mapping_report.tsv'}")
    print(f"Wrote {outdir / 'label_unify_target_coverage.tsv'}")
    print(f"Wrote {outdir / 'label_unify_unmatched_labels.tsv'}")

if __name__ == "__main__":
    main()
