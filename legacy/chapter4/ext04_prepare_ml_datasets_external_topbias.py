#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, gzip, json, math, shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from scipy import sparse


def log(msg: str):
    print(msg, flush=True)


def open_tsv(fp: Path):
    if str(fp).endswith(".gz"):
        return gzip.open(fp, "rt", encoding="utf-8", errors="replace", newline="")
    return open(fp, "rt", encoding="utf-8", errors="replace", newline="")


def write_tsv_gz(fp: Path, header: List[str], rows: List[List[str]]):
    fp.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(fp, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def norm_col(s: str) -> str:
    if s is None:
        return ""
    x = str(s).replace("\ufeff", "").strip()
    x = x.lstrip("#").strip()
    return x.lower()


def detect_mode(training_dir: Path) -> str:
    if list(training_dir.glob("*.train.tsv.gz")) or list(training_dir.glob("*.train.tsv")):
        return "split_files"
    return "split_col"


def iter_rows_split_files(training_dir: Path):
    for split in ["train", "val", "test"]:
        fps = sorted(list(training_dir.glob(f"*.{split}.tsv.gz")) + list(training_dir.glob(f"*.{split}.tsv")))
        for fp in fps:
            with open_tsv(fp) as f:
                r = csv.reader(f, delimiter="\t")
                header = next(r)
                idx = {c: i for i, c in enumerate(header)}
                for a in r:
                    d = {c: (a[i] if i < len(a) else "") for c, i in idx.items()}
                    yield split, d


def iter_rows_split_col(training_dir: Path):
    fps = sorted(list(training_dir.glob("*.tsv.gz")) + list(training_dir.glob("*.tsv")))
    for fp in fps:
        with open_tsv(fp) as f:
            r = csv.reader(f, delimiter="\t")
            header = next(r)
            idx = {c: i for i, c in enumerate(header)}
            if "split" not in idx:
                raise RuntimeError(f"[BAD] split_col mode needs 'split' column, but not found in {fp}")
            for a in r:
                sp = a[idx["split"]] if idx["split"] < len(a) else ""
                if sp not in ("train", "val", "test"):
                    continue
                d = {c: (a[i] if i < len(a) else "") for c, i in idx.items()}
                yield sp, d


def parse_frac_tag(tag: str) -> float:
    if tag == "top1p":
        return 0.01
    if tag == "top0p5p":
        return 0.005
    if tag == "top0p1p":
        return 0.001
    raise ValueError(f"[BAD] unsupported tag: {tag}")


def kth_threshold_abs(values: List[float], frac: float) -> float:
    n = len(values)
    if n == 0:
        return float("inf")
    k = max(1, int(math.ceil(frac * n)))
    vals = np.asarray(values, dtype=np.float64)
    thr = np.partition(-vals, k - 1)[k - 1]
    return float(-thr)


_DECODER = json.JSONDecoder()


def parse_kmer_json(s: str) -> Dict[str, float]:
    if s is None:
        return {}
    x = str(s).strip()
    if x in ("", "NA", "NaN", "nan", "None", "null", "{}"):
        return {}
    try:
        obj, _ = _DECODER.raw_decode(x)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
            return out
        return {}
    except Exception:
        return {}


def read_union_vocab(union_tsv: Path, min_k: int, max_k: int, top_per_k: int, cap_total: int) -> List[Tuple[int, str]]:
    if not union_tsv.exists():
        raise FileNotFoundError(f"[BAD] missing union vocab: {union_tsv}")
    rows = []
    with open_tsv(union_tsv) as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        idx = {c: i for i, c in enumerate(header)}
        if "k" not in idx or "kmer" not in idx:
            raise RuntimeError(f"[BAD] union vocab missing k/kmer columns: {union_tsv}")
        has_score = "score" in idx
        has_df = "df" in idx
        for a in r:
            k = int(a[idx["k"]])
            if k < min_k or k > max_k:
                continue
            mer = a[idx["kmer"]]
            score = float(a[idx["score"]]) if has_score and idx["score"] < len(a) else 0.0
            df = float(a[idx["df"]]) if has_df and idx["df"] < len(a) else 0.0
            rows.append((k, mer, score, df))

    out = []
    for k in range(min_k, max_k + 1):
        kk = [x for x in rows if x[0] == k]
        kk.sort(key=lambda x: (x[2], x[3]), reverse=True)
        if top_per_k > 0:
            kk = kk[:top_per_k]
        out.extend(kk)

    out.sort(key=lambda x: (x[2], x[3]), reverse=True)
    if cap_total > 0:
        out = out[:cap_total]
    return [(k, mer) for (k, mer, _, _) in out]


def choose_kmer_regions_dir(feature_dir: Path, override_dir: Optional[Path], override_col: Optional[str]) -> Tuple[Path, str]:
    if override_dir is not None:
        if not override_dir.exists():
            raise FileNotFoundError(f"[BAD] --kmer_regions_dir not found: {override_dir}")
        col = override_col or "kmer_json"
        return override_dir, col

    direct = feature_dir / "kmer_region_chunks"
    if direct.exists():
        return direct, "kmer_json"

    candidates = []
    for fp in feature_dir.rglob("*.tsv.gz"):
        try:
            with open_tsv(fp) as f:
                line = f.readline()
            cols = [norm_col(x) for x in line.rstrip("\n").split("\t")]
            has_sid = ("seq_id" in cols) or ("seqid" in cols)
            has_kj = ("kmer_json" in cols) or any(("kmer" in c and "json" in c) for c in cols)
            if has_sid and has_kj:
                candidates.append(fp)
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"[BAD] cannot find any kmer regions files under {feature_dir}. Try --kmer_regions_dir explicitly.")
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0].parent, "kmer_json"


def load_kmer_map(regions_dir: Path, kmer_col: str, needed: Set[str],
                  key2idx: Dict[str, int], debug_dir: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, int]]:
    fps = sorted(list(regions_dir.glob("*.tsv.gz")))
    if not fps:
        raise FileNotFoundError(f"[BAD] no *.tsv.gz under kmer regions dir: {regions_dir}")

    out: Dict[str, Dict[str, float]] = {}
    left = set(needed)

    audit = {
        "dict_nonempty": 0,
        "dict_empty": 0,
        "dict_has_keys_but_no_match": 0,
        "dict_has_match": 0,
    }
    key_samples = []

    for fp in fps:
        if not left:
            break
        with open_tsv(fp) as f:
            r = csv.reader(f, delimiter="\t")
            header_raw = next(r)
            header_norm = [norm_col(x) for x in header_raw]
            n2i = {c: i for i, c in enumerate(header_norm)}

            sid_i = None
            for cand in ("seq_id", "seqid"):
                if cand in n2i:
                    sid_i = n2i[cand]
                    break
            if sid_i is None:
                raise RuntimeError(f"[BAD] missing Seq_ID column in {fp}. header={header_raw}")

            kmer_i = None
            kc = norm_col(kmer_col)
            if kc in n2i:
                kmer_i = n2i[kc]
            else:
                for c, i in n2i.items():
                    if ("kmer" in c) and ("json" in c):
                        kmer_i = i
                        break
            if kmer_i is None:
                raise RuntimeError(f"[BAD] missing kmer json column in {fp}. header={header_raw}")

            for a in r:
                if not left:
                    break
                if sid_i >= len(a):
                    continue
                sid = a[sid_i]
                if sid not in left:
                    continue

                raw = a[kmer_i] if kmer_i < len(a) else ""
                d = parse_kmer_json(raw)

                if not d:
                    audit["dict_empty"] += 1
                    out[sid] = {}
                    left.remove(sid)
                    continue

                audit["dict_nonempty"] += 1
                any_match = any((kk in key2idx) for kk in d.keys())
                if any_match:
                    audit["dict_has_match"] += 1
                else:
                    audit["dict_has_keys_but_no_match"] += 1
                    if len(key_samples) < 50:
                        key_samples.append([sid, next(iter(d.keys()))])

                out[sid] = d
                left.remove(sid)

    for sid in left:
        out[sid] = {}
        audit["dict_empty"] += 1

    debug_dir.mkdir(parents=True, exist_ok=True)
    if key_samples:
        write_tsv_gz(debug_dir / "kmer_key_samples_no_vocab_match.tsv.gz",
                     ["Seq_ID", "sample_key"], key_samples)

    return out, audit


def build_sparse_core(rows: List[dict], feat_cols: List[str]) -> sparse.csr_matrix:
    X = np.zeros((len(rows), len(feat_cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(feat_cols):
            v = r.get(c, "")
            if v in ("", "NA", "NaN", "nan", "None", "null"):
                X[i, j] = 0.0
            else:
                try:
                    X[i, j] = float(v)
                except Exception:
                    X[i, j] = 0.0
    return sparse.csr_matrix(X)


def build_sparse_kmer(rows: List[dict], kmer_map: Dict[str, Dict[str, float]], key2idx: Dict[str, int], x_mode: str) -> sparse.csr_matrix:
    data, ri, ci = [], [], []
    for i, r in enumerate(rows):
        sid = r["Seq_ID"]
        d = kmer_map.get(sid, {})
        if not d:
            continue
        for k, v in d.items():
            j = key2idx.get(k)
            if j is None:
                continue
            vv = 1.0 if x_mode == "presence" else float(v)
            if vv == 0:
                continue
            data.append(vv); ri.append(i); ci.append(j)
    return sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(ri, dtype=np.int32), np.asarray(ci, dtype=np.int32))),
        shape=(len(rows), len(key2idx)),
    ).tocsr()


def save_variant(out_dir: Path,
                 Xtr, Xva, Xte,
                 ytr, yva, yte,
                 feature_names: List[str],
                 splits: Dict[str, List[dict]],
                 meta_cols: List[str],
                 tag: str, compare_id: str,
                 args, extra_info: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(out_dir / "X_train.npz", Xtr)
    sparse.save_npz(out_dir / "X_val.npz", Xva)
    sparse.save_npz(out_dir / "X_test.npz", Xte)
    np.save(out_dir / "y_train.npy", ytr)
    np.save(out_dir / "y_val.npy", yva)
    np.save(out_dir / "y_test.npy", yte)
    (out_dir / "feature_names.tsv").write_text("\n".join(feature_names) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True)
    ap.add_argument("--weaver_subdir", required=True)
    ap.add_argument("--feature_subdir", required=True)
    ap.add_argument("--out_subdir", required=True)
    ap.add_argument("--tags", required=True)

    ap.add_argument("--y_col", default="log2fc")
    ap.add_argument("--y_clip", type=float, default=6.0)

    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=3)

    ap.add_argument("--min_k", type=int, default=6)
    ap.add_argument("--max_k", type=int, default=8)
    ap.add_argument("--kmer_x_mode", choices=["count", "presence"], default="presence")
    ap.add_argument("--top_per_k", type=int, default=200)
    ap.add_argument("--kmer_cap_total", type=int, default=1600)
    ap.add_argument("--kmer_union_tsv", default=None)

    ap.add_argument("--kmer_regions_dir", default=None)
    ap.add_argument("--kmer_json_col", default="kmer_json")

    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    proj = Path(args.project_dir)
    tr_dir = proj / args.weaver_subdir / "training_chunks"
    feature_dir = proj / args.feature_subdir
    out_root = proj / args.out_subdir

    if not tr_dir.exists():
        raise FileNotFoundError(f"[BAD] missing training_chunks: {tr_dir}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"[BAD] missing feature_subdir: {feature_dir}")

    if out_root.exists() and args.force:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for t in tags:
        parse_frac_tag(t)

    mode = detect_mode(tr_dir)
    it = iter_rows_split_files(tr_dir) if mode == "split_files" else iter_rows_split_col(tr_dir)
    log(f"[INFO] training_chunks mode = {mode}")

    absY: Dict[str, List[float]] = {}
    nscan = 0
    for sp, r in it:
        if "compare_id" not in r or "Seq_ID" not in r:
            raise RuntimeError("[BAD] training chunks must have compare_id and Seq_ID columns.")
        try:
            y = float(r.get(args.y_col, "0") or 0)
        except Exception:
            y = 0.0
        absY.setdefault(r["compare_id"], []).append(abs(y))
        nscan += 1
    log(f"[INFO] scanned rows = {nscan}, compare_id = {len(absY)}")

    thresholds = {tag: {cid: kth_threshold_abs(vals, parse_frac_tag(tag)) for cid, vals in absY.items()} for tag in tags}
    log("[INFO] thresholds built for tags")

    it2 = iter_rows_split_files(tr_dir) if mode == "split_files" else iter_rows_split_col(tr_dir)
    _, first = next(it2)
    all_cols = list(first.keys())
    feat_cols = [c for c in all_cols if c.startswith("feat_")]
    log(f"[INFO] core feat_cols={len(feat_cols)}")

    union_tsv = Path(args.kmer_union_tsv) if args.kmer_union_tsv else (feature_dir / "kmervocab" / "kmer_union.tsv")
    chosen = read_union_vocab(union_tsv, args.min_k, args.max_k, args.top_per_k, args.kmer_cap_total)
    regions = [f"head{args.head_win}"] + [f"mid{i}" for i in range(args.mid_bins)] + [f"tail{args.tail_win}"]
    raw_keys = [f"k{k}_{reg}_{mer}" for (k, mer) in chosen for reg in regions]
    key2idx = {k: i for i, k in enumerate(raw_keys)}
    log(f"[INFO] kmer vocab: kmers={len(chosen)} region_keys={len(raw_keys)} k_range={args.min_k}-{args.max_k}")

    override_dir = Path(args.kmer_regions_dir) if args.kmer_regions_dir else None
    regions_dir, kmer_col = choose_kmer_regions_dir(feature_dir, override_dir, args.kmer_json_col)
    log(f"[INFO] kmer regions source: dir={regions_dir} col={kmer_col}")

    debug_dir = out_root / "_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        bucket = {cid: {"train": [], "val": [], "test": []} for cid in absY.keys()}
        it3 = iter_rows_split_files(tr_dir) if mode == "split_files" else iter_rows_split_col(tr_dir)

        kept = 0
        for sp, r in it3:
            cid = r["compare_id"]
            try:
                y = float(r.get(args.y_col, "0") or 0)
            except Exception:
                y = 0.0
            if abs(y) < thresholds[tag][cid]:
                continue
            r["_y_clipped"] = float(np.clip(y, -args.y_clip, args.y_clip))
            bucket[cid][sp].append(r)
            kept += 1
        log(f"[INFO] tag={tag} kept_rows={kept}")

        needed: Set[str] = set()
        for cid, spd in bucket.items():
            for sp in ["train", "val", "test"]:
                needed.update(rr["Seq_ID"] for rr in spd[sp])
        log(f"[INFO] tag={tag} needed Seq_ID = {len(needed)}")

        kmer_map, audit = load_kmer_map(regions_dir, kmer_col, needed, key2idx, debug_dir)
        log(f"[INFO] tag={tag} kmer_audit: {audit}")

        # 这里只做数据集输出（你后续训练脚本用它）
        tag_root = out_root / tag
        tag_root.mkdir(parents=True, exist_ok=True)

        for cid, spd in bucket.items():
            Xtr_core = build_sparse_core(spd["train"], feat_cols)
            Xva_core = build_sparse_core(spd["val"], feat_cols)
            Xte_core = build_sparse_core(spd["test"], feat_cols)
            ytr = np.asarray([rr["_y_clipped"] for rr in spd["train"]], dtype=np.float32)
            yva = np.asarray([rr["_y_clipped"] for rr in spd["val"]], dtype=np.float32)
            yte = np.asarray([rr["_y_clipped"] for rr in spd["test"]], dtype=np.float32)

            Xtr_k = build_sparse_kmer(spd["train"], kmer_map, key2idx, args.kmer_x_mode)
            Xva_k = build_sparse_kmer(spd["val"], kmer_map, key2idx, args.kmer_x_mode)
            Xte_k = build_sparse_kmer(spd["test"], kmer_map, key2idx, args.kmer_x_mode)

            cid_root = tag_root / cid
            save_variant(cid_root / "no_kmer", Xtr_core, Xva_core, Xte_core, ytr, yva, yte, feat_cols, spd, [], tag, cid, args, {})
            save_variant(cid_root / "kmer_only", Xtr_k, Xva_k, Xte_k, ytr, yva, yte, list(key2idx.keys()), spd, [], tag, cid, args, {})
            Xtr_all = sparse.hstack([Xtr_core, Xtr_k], format="csr")
            Xva_all = sparse.hstack([Xva_core, Xva_k], format="csr")
            Xte_all = sparse.hstack([Xte_core, Xte_k], format="csr")
            save_variant(cid_root / "all", Xtr_all, Xva_all, Xte_all, ytr, yva, yte, feat_cols + list(key2idx.keys()), spd, [], tag, cid, args, {})

        log(f"[DONE] tag={tag}")

    log(f"[DONE] all tags -> {out_root}")


if __name__ == "__main__":
    main()
