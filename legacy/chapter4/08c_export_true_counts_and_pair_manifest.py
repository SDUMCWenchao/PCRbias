#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import csv
import gzip
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


PREFERRED_VARIANTS = [
    "no_kmer_noprimer",
    "no_kmer",
    "all_noprimer",
    "all",
    "real_kmer_only_all",
    "kmer_only_all",
]


def open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def find_meta_file(d: Path, split: str) -> Path | None:
    # meta_train.tsv.gz / meta_train.tsv
    for ext in [".tsv.gz", ".tsv"]:
        p = d / f"meta_{split}{ext}"
        if p.exists():
            return p
    return None


def find_y_file(d: Path, split: str) -> Path | None:
    # y_train.npy / y_train.npz
    for ext in [".npy", ".npz"]:
        p = d / f"y_{split}{ext}"
        if p.exists():
            return p
    return None


def load_y(path: Path) -> np.ndarray | None:
    if path is None or (not path.exists()):
        return None
    if str(path).endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if str(path).endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        # try common keys
        for k in ["y", "arr_0", "Y"]:
            if k in z.files:
                return z[k]
        # fallback: first array
        if len(z.files) > 0:
            return z[z.files[0]]
    return None


def meta_counts(meta_path: Path):
    """
    Return:
      n_rows (excluding header),
      uniq_pair_id count,
      uniq_yes_file_id count,
      uniq_no_file_id count,
      a dict {pair_id: (yes_file_id, no_file_id)} from this split (for manifest)
    """
    n = 0
    pair_set = set()
    yes_set = set()
    no_set = set()
    pair_map = {}  # pair_id -> (yes_file_id, no_file_id)

    with open_text(meta_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = ["pair_id", "yes_file_id", "no_file_id"]
        for r in required:
            if r not in reader.fieldnames:
                raise RuntimeError(f"Missing column {r} in {meta_path}. Header={reader.fieldnames}")

        for row in reader:
            n += 1
            pid = row["pair_id"]
            yid = row["yes_file_id"]
            nid = row["no_file_id"]
            pair_set.add(pid)
            yes_set.add(yid)
            no_set.add(nid)
            # keep first mapping
            if pid not in pair_map:
                pair_map[pid] = (yid, nid)
            else:
                # sanity check consistency
                if pair_map[pid] != (yid, nid):
                    # same pair_id but different mapping inside same split => abnormal
                    pass

    return n, len(pair_set), len(yes_set), len(no_set), pair_map


def iter_dataset_dirs(inputs_root: Path):
    """
    Yield (tag, species, locus, variant, dirpath) for dirs like:
      inputs_root/tag/species/locus/variant/
    Must contain at least meta_train.*
    """
    for tag_dir in sorted(inputs_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        for sp_dir in sorted(tag_dir.iterdir()):
            if not sp_dir.is_dir():
                continue
            species = sp_dir.name
            for loc_dir in sorted(sp_dir.iterdir()):
                if not loc_dir.is_dir():
                    continue
                locus = loc_dir.name
                for var_dir in sorted(loc_dir.iterdir()):
                    if not var_dir.is_dir():
                        continue
                    variant = var_dir.name
                    if find_meta_file(var_dir, "train") is not None:
                        yield tag, species, locus, variant, var_dir


def choose_reference_variant(variants: list[str]) -> str:
    for v in PREFERRED_VARIANTS:
        if v in variants:
            return v
    return sorted(variants)[0]


def read_samples_meta(samples_meta: Path):
    """
    Build mapping: file_id -> meta_row(dict)
    Accept gz or plain.
    """
    if not samples_meta.exists():
        raise FileNotFoundError(samples_meta)

    m = {}
    with open_text(samples_meta) as f:
        reader = csv.DictReader(f, delimiter="\t")
        # guess key column
        if "file_id" in reader.fieldnames:
            keycol = "file_id"
        elif "File_ID" in reader.fieldnames:
            keycol = "File_ID"
        else:
            raise RuntimeError(f"Cannot find file_id column in {samples_meta}. Header={reader.fieldnames}")

        for row in reader:
            fid = row.get(keycol)
            if fid is None:
                continue
            m[fid] = row
    return m


def pick_sample_name(row: dict) -> str:
    for k in ["sample_name", "SampleName", "sample", "Sample", "name", "Name"]:
        if k in row and row[k]:
            return row[k]
    # fallback to file_id
    return row.get("file_id", row.get("File_ID", ""))


def main():
    ap = argparse.ArgumentParser(
        description="Export true counts (no scaling) and pair_manifest for Results writing."
    )
    ap.add_argument("--project_dir", default=".", help="Project root (default: .)")
    ap.add_argument("--inputs_root", default=None,
                    help="Root of model inputs. Default: <project_dir>/analysis_results/05_ModelInputs_v3_topbias")
    ap.add_argument("--samples_meta", default=None,
                    help="samples_meta.tsv path. Default: <project_dir>/samples_meta.tsv")
    ap.add_argument("--out_dir", default=None,
                    help="Output dir. Default: <project_dir>/analysis_results/07_PlotTables_v3_topbias")
    ap.add_argument("--y_clip", type=float, default=6.0, help="y_clip used in dataset building (default 6)")
    args = ap.parse_args()

    project_dir = Path(args.project_dir).resolve()
    inputs_root = Path(args.inputs_root).resolve() if args.inputs_root else (project_dir / "analysis_results/05_ModelInputs_v3_topbias")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (project_dir / "analysis_results/07_PlotTables_v3_topbias")
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_meta = Path(args.samples_meta).resolve() if args.samples_meta else (project_dir / "samples_meta.tsv")
    samples_map = read_samples_meta(samples_meta)

    # collect all dataset dirs
    groups = defaultdict(list)  # (tag,species,locus) -> list of (variant, dir)
    for tag, sp, loc, var, d in iter_dataset_dirs(inputs_root):
        groups[(tag, sp, loc)].append((var, d))

    if not groups:
        print(f"[ERROR] No datasets found under {inputs_root}", file=sys.stderr)
        sys.exit(2)

    # 1) TRUE COUNTS (reference variant per group)
    true_counts_rows = []
    true_counts_all_variants_rows = []

    # 2) PAIR MANIFEST (from reference variant)
    pair_manifest_rows = []

    for (tag, sp, loc), items in sorted(groups.items()):
        variants = [v for v, _ in items]
        ref_var = choose_reference_variant(variants)
        ref_dir = dict(items)[ref_var]

        # for all variants: counts only (fast)
        for v, d in items:
            for split in ["train", "val", "test"]:
                meta = find_meta_file(d, split)
                if meta is None:
                    continue
                try:
                    n_rows, n_pairs, n_yes, n_no, _ = meta_counts(meta)
                except Exception as e:
                    print(f"[WARN] failed counting {tag}/{sp}/{loc}/{v} split={split}: {e}", file=sys.stderr)
                    continue
                true_counts_all_variants_rows.append({
                    "tag": tag, "species": sp, "locus": loc, "variant": v, "split": split,
                    "n_rows": n_rows,
                    "n_unique_pair_id": n_pairs,
                    "n_unique_yes_file_id": n_yes,
                    "n_unique_no_file_id": n_no,
                    "meta_path": str(meta),
                })

        # reference variant: counts + y stats + pair mapping
        pair_map_union = {}  # pair_id -> (yes_file_id, no_file_id)
        for split in ["train", "val", "test"]:
            meta = find_meta_file(ref_dir, split)
            if meta is None:
                print(f"[WARN] missing meta_{split} for {tag}/{sp}/{loc}/{ref_var}", file=sys.stderr)
                continue
            n_rows, n_pairs, n_yes, n_no, pair_map = meta_counts(meta)
            # merge pair maps
            for pid, yn in pair_map.items():
                if pid not in pair_map_union:
                    pair_map_union[pid] = yn

            # y stats
            y_path = find_y_file(ref_dir, split)
            y = load_y(y_path) if y_path else None
            y_n = y.size if y is not None else ""
            y_mean = float(np.mean(y)) if y is not None else ""
            y_std = float(np.std(y)) if y is not None else ""
            y_q05 = float(np.quantile(y, 0.05)) if y is not None else ""
            y_q25 = float(np.quantile(y, 0.25)) if y is not None else ""
            y_q50 = float(np.quantile(y, 0.50)) if y is not None else ""
            y_q75 = float(np.quantile(y, 0.75)) if y is not None else ""
            y_q95 = float(np.quantile(y, 0.95)) if y is not None else ""
            y_min = float(np.min(y)) if y is not None else ""
            y_max = float(np.max(y)) if y is not None else ""
            if y is not None:
                frac_pos = float(np.mean(y > 0))
                frac_neg = float(np.mean(y < 0))
                frac_zero = float(np.mean(y == 0))
                frac_clip_pos = float(np.mean(y >= args.y_clip))
                frac_clip_neg = float(np.mean(y <= -args.y_clip))
            else:
                frac_pos = frac_neg = frac_zero = frac_clip_pos = frac_clip_neg = ""

            true_counts_rows.append({
                "tag": tag, "species": sp, "locus": loc,
                "ref_variant": ref_var, "split": split,
                "n_rows": n_rows,
                "n_unique_pair_id": n_pairs,
                "n_unique_yes_file_id": n_yes,
                "n_unique_no_file_id": n_no,
                "y_n": y_n,
                "y_mean": y_mean, "y_std": y_std,
                "y_q05": y_q05, "y_q25": y_q25, "y_q50": y_q50, "y_q75": y_q75, "y_q95": y_q95,
                "y_min": y_min, "y_max": y_max,
                "y_frac_pos": frac_pos, "y_frac_neg": frac_neg, "y_frac_zero": frac_zero,
                "y_frac_clip_pos": frac_clip_pos, "y_frac_clip_neg": frac_clip_neg,
                "meta_path": str(meta),
                "y_path": str(y_path) if y_path else "",
            })

        # build pair manifest from union map
        for pid, (yes_fid, no_fid) in sorted(pair_map_union.items()):
            yes_meta = samples_map.get(yes_fid, {})
            no_meta = samples_map.get(no_fid, {})

            yes_name = pick_sample_name(yes_meta) if yes_meta else ""
            no_name = pick_sample_name(no_meta) if no_meta else ""

            # pull common fields if exist
            def g(row, key, default=""):
                return row.get(key, default) if row else default

            # try common PCR / n_individuals keys
            def get_pcr(row):
                for k in ["pcr", "PCR", "is_pcr", "pcr_status"]:
                    if k in row and row[k] != "":
                        return row[k]
                return ""

            def get_nind(row):
                for k in ["n_individuals", "N_individuals", "nInd", "n_ind"]:
                    if k in row and row[k] != "":
                        return row[k]
                return ""

            # sanity checks (soft)
            warn_msgs = []
            if yes_meta and no_meta:
                # should match species/locus/n_individuals, differ only by pcr
                ys = g(yes_meta, "species", "")
                ns = g(no_meta, "species", "")
                yl = g(yes_meta, "locus", "")
                nl = g(no_meta, "locus", "")
                if ys and ns and ys != ns:
                    warn_msgs.append(f"species mismatch yes={ys} no={ns}")
                if yl and nl and yl != nl:
                    warn_msgs.append(f"locus mismatch yes={yl} no={nl}")
                yp = get_pcr(yes_meta)
                npcr = get_pcr(no_meta)
                if yp and npcr and yp == npcr:
                    warn_msgs.append(f"pcr not different (both {yp})")

            pair_manifest_rows.append({
                "tag": tag, "species": sp, "locus": loc,
                "pair_id": pid,
                "yes_file_id": yes_fid,
                "yes_sample_name": yes_name,
                "yes_pcr": get_pcr(yes_meta),
                "yes_n_individuals": get_nind(yes_meta),
                "no_file_id": no_fid,
                "no_sample_name": no_name,
                "no_pcr": get_pcr(no_meta),
                "no_n_individuals": get_nind(no_meta),
                "notes": ";".join(warn_msgs),
            })

    # write outputs
    def write_tsv(path: Path, rows: list[dict]):
        if not rows:
            print(f"[WARN] empty rows for {path}", file=sys.stderr)
            return
        cols = list(rows[0].keys())
        with open(path, "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    out_true = out_dir / "true_counts_by_group.tsv"
    out_allv = out_dir / "true_counts_all_variants.tsv"
    out_pair = out_dir / "pair_manifest.tsv"

    write_tsv(out_true, true_counts_rows)
    write_tsv(out_allv, true_counts_all_variants_rows)
    write_tsv(out_pair, pair_manifest_rows)

    print(f"[DONE] true counts (ref variant) -> {out_true}")
    print(f"[DONE] true counts (all variants) -> {out_allv}")
    print(f"[DONE] pair manifest -> {out_pair}")

    # quick summary
    print(f"[INFO] groups = {len(groups)}  rows(true_counts_by_group)={len(true_counts_rows)}  pairs={len(pair_manifest_rows)}")


if __name__ == "__main__":
    main()
