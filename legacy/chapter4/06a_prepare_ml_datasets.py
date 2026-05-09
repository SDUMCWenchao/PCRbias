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


def stable_split_for_pair_ids(pair_ids: np.ndarray, train_frac: float, val_frac: float):
    """
    Vectorized-ish split by hashing unique pair_id.
    Returns an array of {"train","val","test"}.
    """
    uniq = pd.unique(pair_ids.astype(str))
    mp = {}
    for pid in uniq:
        h = hashlib.md5(pid.encode("utf-8")).hexdigest()
        x = int(h[:8], 16) / 0xFFFFFFFF
        if x < train_frac:
            mp[pid] = "train"
        elif x < train_frac + val_frac:
            mp[pid] = "val"
        else:
            mp[pid] = "test"
    return np.array([mp[str(x)] for x in pair_ids], dtype=object)


def parse_report_top_features(report_md: Path, group="ALL = ALL", topn=3000) -> list[str]:
    in_group = False
    in_table = False
    feats = []
    with report_md.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            m = RE_GROUP.match(line)
            if m:
                in_group = (m.group(1).strip() == group)
                in_table = False
                continue
            if not in_group:
                continue
            if line.startswith("|rank|feature|joint_stable|"):
                in_table = True
                continue
            if not in_table:
                continue
            if line.startswith("|---"):
                continue
            if not line.startswith("|"):
                continue
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) < 2 or not parts[0].isdigit():
                continue
            feats.append(parts[1])
            if len(feats) >= topn:
                break
    return feats


def is_kmer_feat(name: str) -> bool:
    return re.match(r"^k\d+_", name) is not None


def iter_fasta_records(fa_path: Path):
    opener = gzip.open if fa_path.name.endswith(".gz") else open
    with opener(fa_path, "rt", encoding="utf-8", errors="replace") as f:
        seq_id = None
        buf = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_id is not None:
                    yield seq_id, "".join(buf).upper()
                seq_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if seq_id is not None:
            yield seq_id, "".join(buf).upper()


def split_mid_region(seq: str, head_win: int, tail_win: int, mid_bins: int):
    L = len(seq)
    head = seq[:min(head_win, L)]
    tail = seq[max(0, L - tail_win):] if tail_win > 0 else ""
    mid = ""
    if L > head_win + tail_win:
        mid = seq[head_win:L - tail_win]

    mids = []
    if mid_bins <= 0:
        mids = []
    elif len(mid) == 0:
        mids = [""] * mid_bins
    else:
        base = len(mid) // mid_bins
        rem = len(mid) % mid_bins
        start = 0
        for i in range(mid_bins):
            add = base + (1 if i < rem else 0)
            mids.append(mid[start:start + add])
            start += add
    return head, mids, tail


def build_region_kmer_maps(kmer_features: list[str]):
    """
    Build mapping:
      region_k_map[(region, k)] = {kmer_string: col_index}
    """
    region_k_map = {}
    for j, feat in enumerate(kmer_features):
        m = re.match(r"^k(\d+)_([^_]+)_(.+)$", feat)
        if not m:
            continue
        k = int(m.group(1))
        region = m.group(2)
        kmer = m.group(3)
        d = region_k_map.setdefault((region, k), {})
        d[kmer] = j
    return region_k_map


def cols_vals_for_sequence(seq: str, region_k_map: dict, head_win: int, tail_win: int, mid_bins: int, x_mode: str):
    """
    Return (cols:list[int], vals:list[float]) for one sequence.
    """
    head, mids, tail = split_mid_region(seq, head_win, tail_win, mid_bins)

    # region strings must match feature naming
    regions = [("head30", head)]
    regions += [(f"mid{i}", mids[i]) for i in range(len(mids))]
    regions += [("tail30", tail)]

    if x_mode == "presence":
        colset = set()
        for region_name, s in regions:
            if not s:
                continue
            sl = len(s)
            # for each k used in this region
            for (r, k), km2col in region_k_map.items():
                if r != region_name:
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
    else:
        cnt = {}
        for region_name, s in regions:
            if not s:
                continue
            sl = len(s)
            for (r, k), km2col in region_k_map.items():
                if r != region_name:
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


def build_kmer_sparse_from_fasta(
    fasta_path: Path,
    seqid_to_rows: dict[str, list[int]],
    n_rows: int,
    kmer_features: list[str],
    head_win: int,
    tail_win: int,
    mid_bins: int,
    x_mode: str = "presence",
):
    """
    Build CSR matrix shape=(n_rows, n_features), correctly handling duplicate Seq_ID:
      Seq_ID -> [row indices]
    """
    n_feat = len(kmer_features)
    region_k_map = build_region_kmer_maps(kmer_features)

    # COO buffers (memory efficient arrays)
    rows_a = array("I")
    cols_a = array("I")
    data_a = array("f")  # float32

    need = set(seqid_to_rows.keys())
    hit = 0

    for sid, seq in iter_fasta_records(fasta_path):
        if sid not in need:
            continue

        row_list = seqid_to_rows[sid]
        cols, vals = cols_vals_for_sequence(seq, region_k_map, head_win, tail_win, mid_bins, x_mode)

        if cols:
            # for each record-row that references this Seq_ID, copy the same kmer features
            for ridx in row_list:
                # ridx is guaranteed < n_rows
                for j, v in zip(cols, vals):
                    rows_a.append(ridx)
                    cols_a.append(j)
                    data_a.append(float(v))
        hit += 1

    # Build sparse matrix
    if len(rows_a) == 0:
        return sparse.csr_matrix((n_rows, n_feat), dtype=np.float32)

    rows = np.frombuffer(rows_a, dtype=np.uint32).astype(np.int64, copy=False)
    cols = np.frombuffer(cols_a, dtype=np.uint32).astype(np.int64, copy=False)
    data = np.frombuffer(data_a, dtype=np.float32)

    X = sparse.coo_matrix((data, (rows, cols)), shape=(n_rows, n_feat), dtype=np.float32).tocsr()
    X.sum_duplicates()
    print(f"[INFO] kmer fasta hits={hit} nnz={X.nnz} shape={X.shape}")
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias")
    ap.add_argument("--training_dir", default="analysis_results/03_DataWeaver/training_chunks")
    ap.add_argument("--fasta", default="analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta")
    ap.add_argument("--out_base", default="analysis_results/05_ModelInputs")

    ap.add_argument("--report_no_kmer", default="analysis_results/04_Stats_NoKmer/joint/report.md")
    ap.add_argument("--report_kmer_only", default="analysis_results/04_Stats_KmerOnly/joint/report.md")
    ap.add_argument("--report_all", default="analysis_results/04_Stats_All/joint/report.md")

    ap.add_argument("--top_no_kmer", type=int, default=300)
    ap.add_argument("--top_kmer_only", type=int, default=3000)
    ap.add_argument("--top_all_non_kmer", type=int, default=300)
    ap.add_argument("--top_all_kmer", type=int, default=3000)

    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=3)
    ap.add_argument("--kmer_x_mode", choices=["presence", "count"], default="presence")

    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--val_frac", type=float, default=0.1)

    ap.add_argument("--max_train", type=int, default=2000000)
    ap.add_argument("--max_val", type=int, default=250000)
    ap.add_argument("--max_test", type=int, default=250000)

    ap.add_argument("--chunk_glob", default="chunk_*.train.tsv.gz")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    project = Path(args.project_dir)
    training_dir = project / args.training_dir
    fasta_path = project / args.fasta
    out_base = project / args.out_base
    out_base.mkdir(parents=True, exist_ok=True)

    files = sorted(training_dir.glob(args.chunk_glob))
    if not files:
        raise SystemExit(f"[ERROR] no training chunk matched: {training_dir}/{args.chunk_glob}")
    print(f"[INFO] training chunks = {len(files)}")

    # detect available columns from first file
    cols0 = pd.read_csv(files[0], sep="\t", compression="gzip", nrows=0).columns.tolist()
    colset0 = set(cols0)

    # ---------- feature lists ----------
    feats_no = parse_report_top_features(project / args.report_no_kmer, topn=args.top_no_kmer * 10)
    feats_no = [f for f in feats_no if (not is_kmer_feat(f)) and (f in colset0)]
    feats_no = feats_no[:args.top_no_kmer]

    feats_k = parse_report_top_features(project / args.report_kmer_only, topn=args.top_kmer_only * 5)
    feats_k = [f for f in feats_k if is_kmer_feat(f)]
    feats_k = feats_k[:args.top_kmer_only]

    feats_all_raw = parse_report_top_features(project / args.report_all, topn=(args.top_all_non_kmer + args.top_all_kmer) * 10)
    all_non = [f for f in feats_all_raw if (not is_kmer_feat(f)) and (f in colset0)]
    all_non = all_non[:args.top_all_non_kmer]
    all_k = feats_k[:args.top_all_kmer]
    feats_all = all_non + all_k

    (out_base / "feature_lists").mkdir(exist_ok=True)
    (out_base / "feature_lists/no_kmer.txt").write_text("\n".join(feats_no) + "\n", encoding="utf-8")
    (out_base / "feature_lists/kmer_only.txt").write_text("\n".join(feats_k) + "\n", encoding="utf-8")
    (out_base / "feature_lists/all.txt").write_text("\n".join(feats_all) + "\n", encoding="utf-8")

    print(f"[INFO] no_kmer features={len(feats_no)}")
    print(f"[INFO] kmer_only features={len(feats_k)}")
    print(f"[INFO] all features={len(feats_all)} (non-kmer {len(all_non)} + kmer {len(all_k)})")

    # ---------- sample rows from training chunks ----------
    rng = np.random.default_rng(args.seed)

    store = {
        "train": {"seq": [], "pair": [], "y": [], "X_no_parts": [], "X_an_parts": []},
        "val":   {"seq": [], "pair": [], "y": [], "X_no_parts": [], "X_an_parts": []},
        "test":  {"seq": [], "pair": [], "y": [], "X_no_parts": [], "X_an_parts": []},
    }

    def remain(split: str):
        lim = {"train": args.max_train, "val": args.max_val, "test": args.max_test}[split]
        return max(0, lim - len(store[split]["y"]))

    base_cols = ["pair_id", "Seq_ID", "log2fc"]
    need_cols = sorted(set(base_cols + feats_no + all_non))

    for fp in files:
        if remain("train") == 0 and remain("val") == 0 and remain("test") == 0:
            break

        df = pd.read_csv(fp, sep="\t", compression="gzip", usecols=lambda c: c in need_cols)

        if not {"pair_id", "Seq_ID", "log2fc"}.issubset(df.columns):
            raise SystemExit(f"[ERROR] missing required cols in {fp.name}")

        splits = stable_split_for_pair_ids(df["pair_id"].to_numpy(), args.train_frac, args.val_frac)
        df["_split"] = splits

        for sp in ["train", "val", "test"]:
            need = remain(sp)
            if need <= 0:
                continue

            sub = df[df["_split"] == sp]
            if sub.empty:
                continue

            if len(sub) > need:
                idx = rng.choice(sub.index.to_numpy(), size=need, replace=False)
                sub = sub.loc[idx]

            # append ids/y
            store[sp]["seq"].extend(sub["Seq_ID"].astype(str).tolist())
            store[sp]["pair"].extend(sub["pair_id"].astype(str).tolist())
            store[sp]["y"].extend(sub["log2fc"].astype(float).tolist())

            if feats_no:
                Xno = sub[feats_no].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
                store[sp]["X_no_parts"].append(Xno)

            if all_non:
                Xan = sub[all_non].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
                store[sp]["X_an_parts"].append(Xan)

    def finalize_dense(sp: str, key_parts: str, n_feat: int):
        parts = store[sp][key_parts]
        if not parts:
            return np.zeros((0, n_feat), dtype=np.float32)
        return np.vstack(parts).astype(np.float32, copy=False)

    for sp in ["train", "val", "test"]:
        store[sp]["y"] = np.asarray(store[sp]["y"], dtype=np.float32)
        store[sp]["X_no"] = finalize_dense(sp, "X_no_parts", len(feats_no))
        store[sp]["X_an"] = finalize_dense(sp, "X_an_parts", len(all_non))
        print(f"[INFO] {sp}: n={len(store[sp]['y'])}")

    # ---------- build Seq_ID -> [row_indices...] (IMPORTANT FIX) ----------
    def build_seqid_to_rows(seq_list: list[str]):
        mp = {}
        for i, sid in enumerate(seq_list):
            mp.setdefault(sid, []).append(i)
        return mp

    seq_rows_train = build_seqid_to_rows(store["train"]["seq"])
    seq_rows_val   = build_seqid_to_rows(store["val"]["seq"])
    seq_rows_test  = build_seqid_to_rows(store["test"]["seq"])

    # ---------- compute kmer sparse ----------
    print("[INFO] computing kmer matrices from fasta ...")
    Xk_train = build_kmer_sparse_from_fasta(
        fasta_path, seq_rows_train, n_rows=len(store["train"]["seq"]), kmer_features=feats_k,
        head_win=args.head_win, tail_win=args.tail_win, mid_bins=args.mid_bins, x_mode=args.kmer_x_mode
    )
    Xk_val = build_kmer_sparse_from_fasta(
        fasta_path, seq_rows_val, n_rows=len(store["val"]["seq"]), kmer_features=feats_k,
        head_win=args.head_win, tail_win=args.tail_win, mid_bins=args.mid_bins, x_mode=args.kmer_x_mode
    )
    Xk_test = build_kmer_sparse_from_fasta(
        fasta_path, seq_rows_test, n_rows=len(store["test"]["seq"]), kmer_features=feats_k,
        head_win=args.head_win, tail_win=args.tail_win, mid_bins=args.mid_bins, x_mode=args.kmer_x_mode
    )

    def dense_to_csr(X):
        return sparse.csr_matrix(X, dtype=np.float32)

    Xa_train = sparse.hstack([dense_to_csr(store["train"]["X_an"]), Xk_train], format="csr", dtype=np.float32)
    Xa_val   = sparse.hstack([dense_to_csr(store["val"]["X_an"]),   Xk_val],   format="csr", dtype=np.float32)
    Xa_test  = sparse.hstack([dense_to_csr(store["test"]["X_an"]),  Xk_test],  format="csr", dtype=np.float32)

    # ---------- save ----------
    def save_variant(name: str, X_train, y_train, X_val, y_val, X_test, y_test, feat_names: list[str], meta: dict):
        out = out_base / name
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "y_train.npy", y_train)
        np.save(out / "y_val.npy", y_val)
        np.save(out / "y_test.npy", y_test)

        (out / "seq_id_train.txt").write_text("\n".join(store["train"]["seq"]) + "\n", encoding="utf-8")
        (out / "seq_id_val.txt").write_text("\n".join(store["val"]["seq"]) + "\n", encoding="utf-8")
        (out / "seq_id_test.txt").write_text("\n".join(store["test"]["seq"]) + "\n", encoding="utf-8")

        (out / "pair_id_train.txt").write_text("\n".join(store["train"]["pair"]) + "\n", encoding="utf-8")
        (out / "pair_id_val.txt").write_text("\n".join(store["val"]["pair"]) + "\n", encoding="utf-8")
        (out / "pair_id_test.txt").write_text("\n".join(store["test"]["pair"]) + "\n", encoding="utf-8")

        (out / "feature_names.txt").write_text("\n".join(feat_names) + "\n", encoding="utf-8")
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if sparse.issparse(X_train):
            sparse.save_npz(out / "X_train.npz", X_train.tocsr())
            sparse.save_npz(out / "X_val.npz", X_val.tocsr())
            sparse.save_npz(out / "X_test.npz", X_test.tocsr())
        else:
            np.save(out / "X_train.npy", X_train)
            np.save(out / "X_val.npy", X_val)
            np.save(out / "X_test.npy", X_test)

        print(f"[DONE] saved -> {out}")

    # no_kmer
    save_variant(
        "no_kmer",
        store["train"]["X_no"], store["train"]["y"],
        store["val"]["X_no"], store["val"]["y"],
        store["test"]["X_no"], store["test"]["y"],
        feats_no,
        meta={"variant": "no_kmer", "n_features": len(feats_no),
              "split": {"train_frac": args.train_frac, "val_frac": args.val_frac}}
    )

    # kmer_only
    save_variant(
        "kmer_only",
        Xk_train, store["train"]["y"],
        Xk_val, store["val"]["y"],
        Xk_test, store["test"]["y"],
        feats_k,
        meta={"variant": "kmer_only", "n_features": len(feats_k),
              "kmer_x_mode": args.kmer_x_mode, "head_win": args.head_win, "tail_win": args.tail_win, "mid_bins": args.mid_bins}
    )

    # all
    save_variant(
        "all",
        Xa_train, store["train"]["y"],
        Xa_val, store["val"]["y"],
        Xa_test, store["test"]["y"],
        all_non + feats_k,
        meta={"variant": "all", "n_features": len(all_non) + len(feats_k),
              "non_kmer_features": len(all_non), "kmer_features": len(feats_k),
              "kmer_x_mode": args.kmer_x_mode, "head_win": args.head_win, "tail_win": args.tail_win, "mid_bins": args.mid_bins}
    )


if __name__ == "__main__":
    main()
