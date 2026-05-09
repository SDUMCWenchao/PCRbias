#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gzip
import hashlib
import json
import re
from array import array
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


RE_GROUP = re.compile(r"^##\s+Group:\s*(.+?)\s*$")


def open_text(p: Path):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if str(p).endswith(".gz") else open(p, "rt", encoding="utf-8", errors="replace")


def stable_u01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def norm_id(x: str) -> str:
    """
    Normalize ids from chunks to match samples_meta file_id/sample_name.
    Handles: paths, extensions, .trim, fastq/fq(.gz), _R1/_R2, etc.
    """
    s = str(x).strip()
    s = s.split("/")[-1]
    s = s.replace(".gz", "")
    for suf in [".fastq", ".fq", ".fasta", ".fa", ".tsv", ".txt", ".csv"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    # common pipeline suffixes
    for suf in [".trim", ".trimmed", ".merged", ".clean", ".filtered"]:
        if s.endswith(suf):
            s = s[: -len(suf)]
    # read suffix
    s = re.sub(r"(_R[12])$", "", s)
    s = re.sub(r"(\.R[12])$", "", s)
    return s


def build_meta_maps(meta: pd.DataFrame):
    """
    Build robust lookup:
      key -> {species,locus,n_individuals,pcr}
    keys include file_id, sample_name, and normalized variants.
    """
    required = {"file_id", "sample_name", "species", "n_individuals", "locus", "pcr"}
    miss = required - set(meta.columns)
    if miss:
        raise SystemExit(f"[ERROR] samples_meta missing columns: {sorted(miss)}")

    key2 = {}
    for _, r in meta.iterrows():
        file_id = str(r["file_id"])
        sample_name = str(r["sample_name"])
        rec = {
            "file_id": file_id,
            "sample_name": sample_name,
            "species": str(r["species"]),
            "locus": str(r["locus"]),
            "n_individuals": int(r["n_individuals"]),
            "pcr": str(r["pcr"]),
        }
        keys = set()
        for k in [file_id, sample_name, norm_id(file_id), norm_id(sample_name)]:
            if k:
                keys.add(k)
        # also add "raw-ish" variants that often appear
        keys.add(file_id.split(".")[0])
        keys.add(sample_name.split(".")[0])
        for k in list(keys):
            if k and k not in key2:
                key2[k] = rec
    return key2


def read_csv_auto(fp: Path, usecols=None):
    comp = "gzip" if str(fp).endswith(".gz") else None
    return pd.read_csv(fp, sep="\t", compression=comp, usecols=usecols)


def parse_group_features(md_path: Path, group: str, max_take: int = 200000) -> list[str]:
    feats = []
    cur = None
    in_table = False
    with open_text(md_path) as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            m = RE_GROUP.match(line.strip())
            if m:
                cur = m.group(1).strip()
                in_table = False
                continue
            if cur != group:
                continue
            if line.startswith("|rank|feature|"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                break
            parts = [x.strip() for x in line.strip("|").split("|")]
            if len(parts) < 2:
                continue
            if not parts[0].isdigit():
                continue
            feats.append(parts[1])
            if len(feats) >= max_take:
                break
    return feats


def k_from_kmerfeat(feat: str) -> int | None:
    m = re.match(r"^k(\d+)_", feat)
    return int(m.group(1)) if m else None


def select_kmers_by_perk(md_path: Path, locus: str, k_list: list[int],
                         top_per_k_all: int, top_per_k_locus: int,
                         cap_total: int) -> list[str]:
    feats_all = parse_group_features(md_path, "ALL = ALL")
    feats_loc = parse_group_features(md_path, f"locus = {locus}")

    def take_per_k(feats, top_per_k):
        out = []
        seen_k = {k: 0 for k in k_list}
        for f in feats:
            k = k_from_kmerfeat(f)
            if k is None or k not in seen_k:
                continue
            if seen_k[k] >= top_per_k:
                continue
            out.append(f)
            seen_k[k] += 1
            if all(seen_k[x] >= top_per_k for x in k_list):
                break
        return out

    cand = take_per_k(feats_all, top_per_k_all) + take_per_k(feats_loc, top_per_k_locus)

    seen = set()
    out = []
    for f in cand:
        if f not in seen:
            seen.add(f)
            out.append(f)
        if len(out) >= cap_total:
            break
    return out


def iter_fasta_records(fa_path: Path):
    with open_text(fa_path) as f:
        sid = None
        buf = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    yield sid, "".join(buf).upper()
                sid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if sid is not None:
            yield sid, "".join(buf).upper()


def split_mid_region(seq: str, head_win: int, tail_win: int, mid_bins: int):
    L = len(seq)
    head = seq[:min(head_win, L)]
    tail = seq[max(0, L - tail_win):] if tail_win > 0 else ""
    mid = seq[head_win:max(head_win, L - tail_win)] if L > head_win + tail_win else ""

    mids = []
    if mid_bins <= 0:
        mids = []
    elif len(mid) == 0:
        mids = [""] * mid_bins
    else:
        base = len(mid) // mid_bins
        rem = len(mid) % mid_bins
        s = 0
        for i in range(mid_bins):
            add = base + (1 if i < rem else 0)
            mids.append(mid[s:s + add])
            s += add
    return head, mids, tail


def build_region_k_map(kmer_features: list[str]):
    pat = re.compile(r"^k(\d+)_(head\d+|tail\d+|mid\d+)_(.+)$")
    rk = {}
    for j, feat in enumerate(kmer_features):
        m = pat.match(feat)
        if not m:
            continue
        k = int(m.group(1))
        region = m.group(2)
        kmer = m.group(3)
        rk.setdefault((region, k), {})[kmer] = j
    return rk


def cols_vals_for_seq(seq: str, rk_map, head_win: int, tail_win: int, mid_bins: int, x_mode: str):
    head, mids, tail = split_mid_region(seq, head_win, tail_win, mid_bins)
    reg_head = f"head{head_win}"
    reg_tail = f"tail{tail_win}"

    segments = [(reg_head, head)] + [(f"mid{i}", mids[i]) for i in range(len(mids))] + [(reg_tail, tail)]

    if x_mode == "presence":
        colset = set()
        for region, s in segments:
            if not s:
                continue
            sl = len(s)
            for (r, k), km2col in rk_map.items():
                if r != region:
                    continue
                if sl < k:
                    continue
                for i in range(0, sl - k + 1):
                    km = s[i:i + k]
                    j = km2col.get(km)
                    if j is not None:
                        colset.add(j)
        cols = sorted(colset)
        vals = [1.0] * len(cols)
        return cols, vals

    cnt = {}
    for region, s in segments:
        if not s:
            continue
        sl = len(s)
        for (r, k), km2col in rk_map.items():
            if r != region:
                continue
            if sl < k:
                continue
            for i in range(0, sl - k + 1):
                km = s[i:i + k]
                j = km2col.get(km)
                if j is not None:
                    cnt[j] = cnt.get(j, 0.0) + 1.0
    cols = sorted(cnt.keys())
    vals = [float(cnt[j]) for j in cols]
    return cols, vals


def build_seqid_to_rows(seq_ids: list[str]):
    mp = {}
    for i, sid in enumerate(seq_ids):
        mp.setdefault(sid, []).append(i)
    return mp


def build_kmer_sparse_three_splits_from_fasta(
    fasta_path: Path,
    seq_ids_train: list[str], seq_ids_val: list[str], seq_ids_test: list[str],
    kmer_features: list[str],
    head_win: int, tail_win: int, mid_bins: int,
    x_mode: str
):
    ntr, nva, nte = len(seq_ids_train), len(seq_ids_val), len(seq_ids_test)
    nf = len(kmer_features)
    rk = build_region_k_map(kmer_features)

    mp_tr = build_seqid_to_rows(seq_ids_train)
    mp_va = build_seqid_to_rows(seq_ids_val)
    mp_te = build_seqid_to_rows(seq_ids_test)
    need = set(mp_tr.keys()) | set(mp_va.keys()) | set(mp_te.keys())

    def buf():
        return array("I"), array("I"), array("f")

    r_tr, c_tr, d_tr = buf()
    r_va, c_va, d_va = buf()
    r_te, c_te, d_te = buf()

    hit = {"train": 0, "val": 0, "test": 0}

    for sid, seq in iter_fasta_records(fasta_path):
        if sid not in need:
            continue
        cols, vals = cols_vals_for_seq(seq, rk, head_win, tail_win, mid_bins, x_mode)

        if sid in mp_tr:
            for ridx in mp_tr[sid]:
                for j, v in zip(cols, vals):
                    r_tr.append(ridx); c_tr.append(j); d_tr.append(float(v))
            hit["train"] += 1
        if sid in mp_va:
            for ridx in mp_va[sid]:
                for j, v in zip(cols, vals):
                    r_va.append(ridx); c_va.append(j); d_va.append(float(v))
            hit["val"] += 1
        if sid in mp_te:
            for ridx in mp_te[sid]:
                for j, v in zip(cols, vals):
                    r_te.append(ridx); c_te.append(j); d_te.append(float(v))
            hit["test"] += 1

    def make(r, c, d, shape):
        if len(d) == 0:
            return sparse.csr_matrix(shape, dtype=np.float32)
        X = sparse.coo_matrix(
            (np.frombuffer(d, dtype=np.float32),
             (np.frombuffer(r, dtype=np.uint32).astype(np.int64, copy=False),
              np.frombuffer(c, dtype=np.uint32).astype(np.int64, copy=False))),
            shape=shape, dtype=np.float32
        ).tocsr()
        X.sum_duplicates()
        if x_mode == "presence":
            X.data[:] = 1.0
        return X

    Xtr = make(r_tr, c_tr, d_tr, (ntr, nf))
    Xva = make(r_va, c_va, d_va, (nva, nf))
    Xte = make(r_te, c_te, d_te, (nte, nf))

    print(f"[INFO] kmer hits train/val/test = {hit['train']}/{hit['val']}/{hit['test']}  nnz = {Xtr.nnz}/{Xva.nnz}/{Xte.nnz}")
    return Xtr, Xva, Xte


def save_variant(out_dir: Path, Xtr, Xva, Xte, ytr, yva, yte, wtr, wva, wte,
                 meta_tr: pd.DataFrame, meta_va: pd.DataFrame, meta_te: pd.DataFrame,
                 feature_names: list[str], config: dict):
    out_dir.mkdir(parents=True, exist_ok=True)

    if sparse.issparse(Xtr) or sparse.issparse(Xva) or sparse.issparse(Xte):
        sparse.save_npz(out_dir / "X_train.npz", Xtr.tocsr())
        sparse.save_npz(out_dir / "X_val.npz", Xva.tocsr())
        sparse.save_npz(out_dir / "X_test.npz", Xte.tocsr())
    else:
        np.save(out_dir / "X_train.npy", Xtr.astype(np.float32, copy=False))
        np.save(out_dir / "X_val.npy", Xva.astype(np.float32, copy=False))
        np.save(out_dir / "X_test.npy", Xte.astype(np.float32, copy=False))

    np.save(out_dir / "y_train.npy", ytr.astype(np.float32, copy=False))
    np.save(out_dir / "y_val.npy", yva.astype(np.float32, copy=False))
    np.save(out_dir / "y_test.npy", yte.astype(np.float32, copy=False))

    np.save(out_dir / "w_train.npy", wtr.astype(np.float32, copy=False))
    np.save(out_dir / "w_val.npy", wva.astype(np.float32, copy=False))
    np.save(out_dir / "w_test.npy", wte.astype(np.float32, copy=False))

    meta_tr.to_csv(out_dir / "meta_train.tsv.gz", sep="\t", index=False, compression="gzip")
    meta_va.to_csv(out_dir / "meta_val.tsv.gz", sep="\t", index=False, compression="gzip")
    meta_te.to_csv(out_dir / "meta_test.tsv.gz", sep="\t", index=False, compression="gzip")

    (out_dir / "feature_names.txt").write_text("\n".join(feature_names) + "\n", encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[DONE] saved -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--training_chunks", default="analysis_results/03_DataWeaver/training_chunks")
    ap.add_argument("--seq_fasta", default="analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta")
    ap.add_argument("--samples_meta", default="scripts/samples_meta.tsv")

    ap.add_argument("--out_root", default="analysis_results/05_ModelInputs_v3")

    ap.add_argument("--species_list", default="donkey,pig,cattle,10mix")
    ap.add_argument("--locus_list", default="12S,16S")

    ap.add_argument("--max_train", type=int, default=1200000)
    ap.add_argument("--max_val", type=int, default=250000)
    ap.add_argument("--max_test", type=int, default=250000)

    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--y_clip", type=float, default=6.0)
    ap.add_argument("--weight_mode", choices=["sqrt_sum", "log1p_sum", "none"], default="sqrt_sum")

    ap.add_argument("--skip_n_individuals_le", type=int, default=1,
                    help="skip rows if yes/no sample has n_individuals <= this (default 1)")

    ap.add_argument("--kmer_report_md", default="analysis_results/04_Stats_KmerOnly/joint/report.md")
    ap.add_argument("--k_list", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--top_per_k_all", type=int, default=200)
    ap.add_argument("--top_per_k_locus", type=int, default=200)
    ap.add_argument("--kmer_cap_total", type=int, default=1600)

    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=3)
    ap.add_argument("--kmer_x_mode", choices=["presence", "count"], default="presence")

    ap.add_argument("--min_pairs_for_pair_split", type=int, default=3,
                    help="if unique pair_id >= this, use pair-level split; otherwise fall back to Seq_ID split")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shuffle_chunks", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="only report pair counts and split mode, then exit")
    args = ap.parse_args()

    project = Path(args.project_dir)
    chunks_dir = project / args.training_chunks
    fasta_path = project / args.seq_fasta
    meta_path = project / args.samples_meta
    out_root = project / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    species_list = [x.strip() for x in args.species_list.split(",") if x.strip()]
    locus_list = [x.strip() for x in args.locus_list.split(",") if x.strip()]
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    meta = pd.read_csv(meta_path, sep="\t")
    key2 = build_meta_maps(meta)

    files = sorted(chunks_dir.glob("chunk_*.train.tsv.gz"))
    if not files:
        files = sorted(chunks_dir.glob("chunk_*.train.tsv"))
    if not files:
        raise SystemExit(f"[ERROR] no chunk files under: {chunks_dir}")
    if args.shuffle_chunks:
        rng.shuffle(files)

    # get header / feat cols
    hdr = read_csv_auto(files[0], usecols=None).columns.tolist()
    required = ["pair_id", "yes_file_id", "no_file_id", "Seq_ID", "count_yes", "count_no", "log2fc"]
    for c in required:
        if c not in hdr:
            raise SystemExit(f"[ERROR] missing col in chunks: {c}")

    feat_cols = [c for c in hdr if c.startswith("feat_")]
    primer_cols = [c for c in feat_cols if c.startswith("feat_pr_")]
    nonprimer_cols = [c for c in feat_cols if not c.startswith("feat_pr_")]
    print(f"[INFO] feat_cols={len(feat_cols)} primer_cols={len(primer_cols)} nonprimer_cols={len(nonprimer_cols)}")

    # ---------- PASS 1: count unique pair_id per (species,locus) after filters ----------
    pair_sets = {(sp, lc): set() for sp in species_list for lc in locus_list}
    dropped_unknown = 0
    dropped_n1 = 0

    for fp in files:
        df = read_csv_auto(fp, usecols=["pair_id", "yes_file_id", "no_file_id"])
        if df.empty:
            continue

        yes_raw = df["yes_file_id"].astype(str).map(norm_id)
        no_raw = df["no_file_id"].astype(str).map(norm_id)

        # lookup meta
        yes_meta = yes_raw.map(lambda k: key2.get(k))
        no_meta = no_raw.map(lambda k: key2.get(k))

        ok = yes_meta.notna() & no_meta.notna()
        dropped_unknown += int((~ok).sum())
        df = df[ok].copy()
        yes_meta = yes_meta[ok]
        no_meta = no_meta[ok]

        # skip n_individuals <= threshold
        yes_n = yes_meta.map(lambda r: r["n_individuals"]).astype(int)
        no_n = no_meta.map(lambda r: r["n_individuals"]).astype(int)
        ok2 = (yes_n > args.skip_n_individuals_le) & (no_n > args.skip_n_individuals_le)
        dropped_n1 += int((~ok2).sum())
        df = df[ok2].copy()
        yes_meta = yes_meta[ok2]

        if df.empty:
            continue

        sp = yes_meta.map(lambda r: r["species"])
        lc = yes_meta.map(lambda r: r["locus"])

        for (spp, lcc), g in df.assign(species=sp.values, locus=lc.values).groupby(["species", "locus"], sort=False):
            if spp in species_list and lcc in locus_list:
                pair_sets[(spp, lcc)].update(g["pair_id"].astype(str).unique().tolist())

    split_mode = {}
    for sp in species_list:
        for lc in locus_list:
            n_pairs = len(pair_sets[(sp, lc)])
            mode = "pair" if n_pairs >= args.min_pairs_for_pair_split else "seq"
            split_mode[(sp, lc)] = mode
            print(f"[INFO] pairs {sp}/{lc}: n_pairs={n_pairs} split_mode={mode}")

    if dropped_unknown:
        print(f"[WARN] dropped rows with unknown file_id mapping (pass1): {dropped_unknown}")
    if dropped_n1:
        print(f"[WARN] dropped rows with n_individuals<= {args.skip_n_individuals_le} (pass1): {dropped_n1}")

    if args.dry_run:
        print("[DONE] dry_run")
        return

    # ---------- PASS 2: build datasets with capped sampling per split ----------
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    counts = {(sp, lc, split): 0 for sp in species_list for lc in locus_list for split in ["train", "val", "test"]}

    def weight(cy, cn):
        if args.weight_mode == "none":
            return 1.0
        s = float(cy + cn)
        if args.weight_mode == "sqrt_sum":
            return float(np.sqrt(max(s, 0.0)))
        return float(np.log1p(max(s, 0.0)))

    def assign_split(mode: str, sp: str, lc: str, pair_id: str, seq_id: str) -> str:
        if mode == "pair":
            u = stable_u01(f"{sp}::{lc}::{pair_id}")
        else:
            u = stable_u01(f"{sp}::{lc}::{seq_id}")
        if u < 0.15:
            return "test"
        if u < 0.30:
            return "val"
        return "train"

    # store blocks for each (sp,lc,split,kind)
    X_blocks = {}
    y_blocks = {}
    w_blocks = {}
    m_blocks = {}

    dropped_unknown2 = 0
    dropped_n1_2 = 0
    dropped_support = 0

    usecols = required + feat_cols
    for fp in files:
        df = read_csv_auto(fp, usecols=usecols)
        if df.empty:
            continue

        # support filter
        support = df["count_yes"].astype(int) + df["count_no"].astype(int)
        ok_sup = support >= args.min_support
        dropped_support += int((~ok_sup).sum())
        df = df[ok_sup].copy()
        if df.empty:
            continue

        yes_norm = df["yes_file_id"].astype(str).map(norm_id)
        no_norm = df["no_file_id"].astype(str).map(norm_id)

        yes_meta = yes_norm.map(lambda k: key2.get(k))
        no_meta = no_norm.map(lambda k: key2.get(k))
        ok = yes_meta.notna() & no_meta.notna()
        dropped_unknown2 += int((~ok).sum())
        df = df[ok].copy()
        yes_meta = yes_meta[ok]
        no_meta = no_meta[ok]
        if df.empty:
            continue

        yes_n = yes_meta.map(lambda r: r["n_individuals"]).astype(int)
        no_n = no_meta.map(lambda r: r["n_individuals"]).astype(int)
        ok2 = (yes_n > args.skip_n_individuals_le) & (no_n > args.skip_n_individuals_le)
        dropped_n1_2 += int((~ok2).sum())
        df = df[ok2].copy()
        yes_meta = yes_meta[ok2]
        if df.empty:
            continue

        df["species"] = yes_meta.map(lambda r: r["species"]).values
        df["locus"] = yes_meta.map(lambda r: r["locus"]).values

        df = df[df["species"].isin(species_list) & df["locus"].isin(locus_list)]
        if df.empty:
            continue

        # y + w + X matrices for this chunk (compute once, then slice)
        y = np.clip(df["log2fc"].astype(float).to_numpy(), -args.y_clip, args.y_clip).astype(np.float32, copy=False)
        w = np.array([weight(int(a), int(b)) for a, b in zip(df["count_yes"].astype(int), df["count_no"].astype(int))], dtype=np.float32)

        X_all = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)
        X_nop = df[nonprimer_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        # group by stratum, then split within
        for (sp, lc), gidx in df.groupby(["species", "locus"], sort=False).groups.items():
            mode = split_mode[(sp, lc)]
            sub = df.loc[gidx, ["Seq_ID", "pair_id", "yes_file_id", "no_file_id"]].copy()
            sub["species"] = sp
            sub["locus"] = lc

            # assign split per row
            splits = []
            for pid, sid in zip(sub["pair_id"].astype(str).tolist(), sub["Seq_ID"].astype(str).tolist()):
                splits.append(assign_split(mode, sp, lc, pid, sid))
            sub["split"] = splits

            # materialize each split with cap
            for split in ["train", "val", "test"]:
                need = caps[split] - counts[(sp, lc, split)]
                if need <= 0:
                    continue
                idx = sub.index[sub["split"] == split].to_numpy()
                if len(idx) == 0:
                    continue
                if len(idx) > need:
                    idx = rng.choice(idx, size=need, replace=False)

                # map to positions in df arrays
                pos = df.index.get_indexer(idx)

                X_blocks.setdefault((sp, lc, split, "all"), []).append(X_all[pos])
                X_blocks.setdefault((sp, lc, split, "noprimer"), []).append(X_nop[pos])
                y_blocks.setdefault((sp, lc, split), []).append(y[pos])
                w_blocks.setdefault((sp, lc, split), []).append(w[pos])
                m_blocks.setdefault((sp, lc, split), []).append(sub.loc[idx].drop(columns=["split"]).reset_index(drop=True))

                counts[(sp, lc, split)] += len(pos)

    if dropped_unknown2:
        print(f"[WARN] dropped rows with unknown file_id mapping (pass2): {dropped_unknown2}")
    if dropped_n1_2:
        print(f"[WARN] dropped rows with n_individuals<= {args.skip_n_individuals_le} (pass2): {dropped_n1_2}")
    if dropped_support:
        print(f"[WARN] dropped rows with support < {args.min_support}: {dropped_support}")

    # finalize each stratum and save
    kmer_md = project / args.kmer_report_md

    for sp in species_list:
        for lc in locus_list:
            ntr = counts[(sp, lc, "train")]
            nva = counts[(sp, lc, "val")]
            nte = counts[(sp, lc, "test")]
            print(f"[INFO] rows {sp}/{lc}: train/val/test = {ntr}/{nva}/{nte}")

            # if still missing (very tiny data), fall back by re-splitting within available rows not implemented here
            if ntr == 0 or nva == 0 or nte == 0:
                print(f"[WARN] skip {sp}/{lc}: still empty split after fallback (train/val/test) -> {ntr}/{nva}/{nte}")
                continue

            def cat_blocks(k, kind=None):
                if kind is None:
                    blocks = y_blocks.get(k, [])
                else:
                    blocks = X_blocks.get((*k, kind), [])
                if not blocks:
                    return None
                return np.concatenate(blocks, axis=0) if isinstance(blocks[0], np.ndarray) else pd.concat(blocks, ignore_index=True)

            # dense X
            Xtr_all = np.concatenate(X_blocks[(sp, lc, "train", "all")], axis=0)
            Xva_all = np.concatenate(X_blocks[(sp, lc, "val", "all")], axis=0)
            Xte_all = np.concatenate(X_blocks[(sp, lc, "test", "all")], axis=0)

            Xtr_nop = np.concatenate(X_blocks[(sp, lc, "train", "noprimer")], axis=0)
            Xva_nop = np.concatenate(X_blocks[(sp, lc, "val", "noprimer")], axis=0)
            Xte_nop = np.concatenate(X_blocks[(sp, lc, "test", "noprimer")], axis=0)

            ytr = np.concatenate(y_blocks[(sp, lc, "train")], axis=0)
            yva = np.concatenate(y_blocks[(sp, lc, "val")], axis=0)
            yte = np.concatenate(y_blocks[(sp, lc, "test")], axis=0)

            wtr = np.concatenate(w_blocks[(sp, lc, "train")], axis=0)
            wva = np.concatenate(w_blocks[(sp, lc, "val")], axis=0)
            wte = np.concatenate(w_blocks[(sp, lc, "test")], axis=0)

            mtr = pd.concat(m_blocks[(sp, lc, "train")], ignore_index=True)
            mva = pd.concat(m_blocks[(sp, lc, "val")], ignore_index=True)
            mte = pd.concat(m_blocks[(sp, lc, "test")], ignore_index=True)

            # select kmers for this locus (ALL + locus, per-k)
            kmers = select_kmers_by_perk(kmer_md, lc, k_list,
                                         args.top_per_k_all, args.top_per_k_locus,
                                         args.kmer_cap_total)

            # compute kmer matrices once per stratum
            seq_tr = mtr["Seq_ID"].astype(str).tolist()
            seq_va = mva["Seq_ID"].astype(str).tolist()
            seq_te = mte["Seq_ID"].astype(str).tolist()

            print(f"[INFO] {sp}/{lc} kmers={len(kmers)} split_mode={split_mode[(sp, lc)]}")
            Xk_tr, Xk_va, Xk_te = build_kmer_sparse_three_splits_from_fasta(
                fasta_path, seq_tr, seq_va, seq_te, kmers,
                args.head_win, args.tail_win, args.mid_bins, args.kmer_x_mode
            )

            base_out = out_root / sp / lc
            cfg_common = dict(
                species=sp, locus=lc,
                head_win=args.head_win, tail_win=args.tail_win, mid_bins=args.mid_bins,
                kmer_x_mode=args.kmer_x_mode,
                min_support=args.min_support,
                y_clip=args.y_clip,
                weight_mode=args.weight_mode,
                skip_n_individuals_le=args.skip_n_individuals_le,
                split_mode=split_mode[(sp, lc)],
                min_pairs_for_pair_split=args.min_pairs_for_pair_split,
                kmer_top_per_k_all=args.top_per_k_all,
                kmer_top_per_k_locus=args.top_per_k_locus,
                kmer_cap_total=args.kmer_cap_total,
                kmer_n=len(kmers),
                n_train=int(len(ytr)), n_val=int(len(yva)), n_test=int(len(yte)),
            )

            # variants
            save_variant(base_out / "no_kmer", Xtr_all, Xva_all, Xte_all, ytr, yva, yte, wtr, wva, wte,
                         mtr, mva, mte, feat_cols, {**cfg_common, "variant": "no_kmer"})

            save_variant(base_out / "no_kmer_noprimer", Xtr_nop, Xva_nop, Xte_nop, ytr, yva, yte, wtr, wva, wte,
                         mtr, mva, mte, nonprimer_cols, {**cfg_common, "variant": "no_kmer_noprimer"})

            save_variant(base_out / "kmer_only_all", Xk_tr, Xk_va, Xk_te, ytr, yva, yte, wtr, wva, wte,
                         mtr, mva, mte, kmers, {**cfg_common, "variant": "kmer_only_all"})

            # slice by k without extra fasta scan
            kmers_by_k = {k: [] for k in k_list}
            cols_by_k = {k: [] for k in k_list}
            for j, f in enumerate(kmers):
                kk = k_from_kmerfeat(f)
                if kk in kmers_by_k:
                    kmers_by_k[kk].append(f)
                    cols_by_k[kk].append(j)

            for kk in k_list:
                cols = cols_by_k[kk]
                if not cols:
                    continue
                save_variant(base_out / f"kmer_only_k{kk}",
                             Xk_tr[:, cols], Xk_va[:, cols], Xk_te[:, cols],
                             ytr, yva, yte, wtr, wva, wte,
                             mtr, mva, mte,
                             kmers_by_k[kk],
                             {**cfg_common, "variant": f"kmer_only_k{kk}", "k": kk})

            # all
            Xmix_tr = sparse.hstack([sparse.csr_matrix(Xtr_all), Xk_tr], format="csr", dtype=np.float32)
            Xmix_va = sparse.hstack([sparse.csr_matrix(Xva_all), Xk_va], format="csr", dtype=np.float32)
            Xmix_te = sparse.hstack([sparse.csr_matrix(Xte_all), Xk_te], format="csr", dtype=np.float32)
            save_variant(base_out / "all", Xmix_tr, Xmix_va, Xmix_te, ytr, yva, yte, wtr, wva, wte,
                         mtr, mva, mte, feat_cols + kmers, {**cfg_common, "variant": "all"})

            Xmix_tr2 = sparse.hstack([sparse.csr_matrix(Xtr_nop), Xk_tr], format="csr", dtype=np.float32)
            Xmix_va2 = sparse.hstack([sparse.csr_matrix(Xva_nop), Xk_va], format="csr", dtype=np.float32)
            Xmix_te2 = sparse.hstack([sparse.csr_matrix(Xte_nop), Xk_te], format="csr", dtype=np.float32)
            save_variant(base_out / "all_noprimer", Xmix_tr2, Xmix_va2, Xmix_te2, ytr, yva, yte, wtr, wva, wte,
                         mtr, mva, mte, nonprimer_cols + kmers, {**cfg_common, "variant": "all_noprimer"})


if __name__ == "__main__":
    main()
