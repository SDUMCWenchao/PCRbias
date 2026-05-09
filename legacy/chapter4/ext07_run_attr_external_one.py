#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, gzip
from pathlib import Path
import numpy as np

def read_header_tsv(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return f.readline().rstrip("\n").split("\t")

def _iter_tsv_candidates(root: Path, split: str):
    pats = [
        f"*{split}*.tsv.gz",
        f"*{split}*.tsv",
        f"*.{split}.tsv.gz",
        f"*.{split}.tsv",
    ]
    seen = set()
    for pat in pats:
        for p in sorted(root.glob(pat)):
            if p in seen: 
                continue
            seen.add(p)
            yield p

def find_split_table_with_seqid(dataset_dir: Path, split: str,
                                project_dir: Path | None = None,
                                weaver_subdir: Path | None = None):
    """
    Priority:
      1) dataset_dir: any *test*.tsv(.gz) containing column Seq_ID
      2) DataWeaver split_files under project_dir/weaver_subdir:
         search with compare_id inferred from dataset_dir path: .../<tag>/<compare_id>/<variant>
    """
    # 1) try dataset_dir local
    for p in _iter_tsv_candidates(dataset_dir, split):
        try:
            hdr = read_header_tsv(p)
        except Exception:
            continue
        if "Seq_ID" in hdr:
            return p, hdr.index("Seq_ID")

    # 2) fallback to DataWeaver split files
    if project_dir is None or weaver_subdir is None:
        raise FileNotFoundError(
            f"[BAD] cannot find any *{split}*.tsv(.gz) with column Seq_ID under {dataset_dir}, "
            f"and no --project_dir/--weaver_subdir provided for fallback."
        )

    # infer compare_id from dataset_dir: .../<tag>/<compare_id>/<variant>
    # dataset_dir = inputs_root/tag/compare_id/variant
    try:
        compare_id = dataset_dir.parent.name
    except Exception:
        compare_id = ""

    weaver_root = project_dir / weaver_subdir
    # common places:
    #   weaver_root/training_chunks/
    #   weaver_root/training_chunks/split_files/
    #   weaver_root/training_chunks/splits/
    search_roots = [
        weaver_root / "training_chunks",
        weaver_root / "training_chunks" / "split_files",
        weaver_root / "training_chunks" / "splits",
        weaver_root,
    ]
    seen = set()
    for sr in search_roots:
        if not sr.exists():
            continue
        # first: narrow by compare_id
        pats = [
            f"*{compare_id}*{split}*.tsv.gz",
            f"*{compare_id}*{split}*.tsv",
        ] if compare_id else [
            f"*{split}*.tsv.gz",
            f"*{split}*.tsv",
        ]
        for pat in pats:
            for p in sorted(sr.rglob(pat)):
                if p in seen:
                    continue
                seen.add(p)
                try:
                    hdr = read_header_tsv(p)
                except Exception:
                    continue
                if "Seq_ID" in hdr:
                    return p, hdr.index("Seq_ID")

    raise FileNotFoundError(
        f"[BAD] cannot find any split table with Seq_ID for split={split}. "
        f"Tried dataset_dir={dataset_dir} and weaver_root={weaver_root} (compare_id={compare_id})."
    )

def load_seq_ids_from_table(path: Path, seq_col_idx: int):
    opener = gzip.open if str(path).endswith(".gz") else open
    ids = []
    with opener(path, "rt") as f:
        _ = f.readline()
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if seq_col_idx >= len(parts):
                continue
            ids.append(parts[seq_col_idx])
    return ids

def load_feature_names(dataset_dir: Path):
    for fn in ["feature_names.txt", "feature_names.tsv", "columns.tsv", "cols.tsv"]:
        p = dataset_dir / fn
        if p.exists():
            return [x.strip().split("\t")[0] for x in p.read_text().splitlines() if x.strip()]
    return None

def load_X(dataset_dir: Path, split: str):
    p_npz = dataset_dir / f"X_{split}.npz"
    p_npy = dataset_dir / f"X_{split}.npy"
    if p_npz.exists():
        from scipy import sparse
        return sparse.load_npz(p_npz)
    if p_npy.exists():
        return np.load(p_npy)
    raise FileNotFoundError(f"[BAD] missing X_{split}.npz/.npy under {dataset_dir}")

def sample_indices(n, explain_n, seed=1):
    if explain_n <= 0 or explain_n >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=explain_n, replace=False)

def write_tsv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

def write_skip_meta(model_dir: Path, model: str, split: str, reason: str, extra: dict | None = None):
    out_dir = model_dir / ("attr_tables" if model == "seqcnn" else "shap_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": model, "split": split, "skipped": True, "reason": reason}
    if extra:
        meta.update(extra)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[SKIP] {model} {split}: {reason} -> {out_dir}/meta.json")

def run_rf_shap(dataset_dir: Path, model_dir: Path, split: str, explain_n: int, seed: int, top_features: int):
    import shap, joblib
    model_path = None
    for fn in ["model.joblib", "model.pkl", "rf.joblib"]:
        p = model_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        write_skip_meta(model_dir, "rf", split, "missing_model_file")
        return

    try:
        X = load_X(dataset_dir, split)
    except Exception as e:
        write_skip_meta(model_dir, "rf", split, f"missing_X_{split}: {e}")
        return

    n = X.shape[0]
    if n == 0:
        write_skip_meta(model_dir, "rf", split, "X has 0 rows")
        return

    idx = sample_indices(n, explain_n, seed)
    if hasattr(X, "tocsr"):
        Xd = X.tocsr()[idx].toarray()
    else:
        Xd = np.asarray(X)[idx]

    rf = joblib.load(model_path)
    explainer = shap.TreeExplainer(rf)
    sv = np.asarray(explainer.shap_values(Xd))

    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_signed = np.mean(sv, axis=0)

    names = load_feature_names(dataset_dir)
    if names is None or len(names) != len(mean_abs):
        names = [f"f{i}" for i in range(len(mean_abs))]

    order = np.argsort(-mean_abs)
    if top_features > 0:
        order = order[:top_features]

    out_dir = model_dir / "shap_tables"
    rows = [(names[i], float(mean_abs[i]), float(mean_signed[i])) for i in order]
    write_tsv(out_dir / f"rf_shap_{split}.tsv", ["feature", "mean_abs_shap", "mean_shap"], rows)
    (out_dir / "meta.json").write_text(json.dumps({"model":"rf","split":split,"n":int(n),"explain_n":int(len(idx))}, indent=2))
    print(f"[DONE] RF SHAP -> {out_dir}")

def run_xgb_contrib(dataset_dir: Path, model_dir: Path, split: str, explain_n: int, seed: int, top_features: int):
    import xgboost as xgb
    model_path = None
    for fn in ["model.json", "model.ubj", "model.bin", "xgb.json", "xgb.ubj"]:
        p = model_dir / fn
        if p.exists():
            model_path = p
            break
    if model_path is None:
        write_skip_meta(model_dir, "xgb", split, "missing_model_file")
        return

    try:
        X = load_X(dataset_dir, split)
    except Exception as e:
        write_skip_meta(model_dir, "xgb", split, f"missing_X_{split}: {e}")
        return

    n = X.shape[0]
    if n == 0:
        write_skip_meta(model_dir, "xgb", split, "X has 0 rows")
        return

    idx = sample_indices(n, explain_n, seed)
    if hasattr(X, "tocsr"):
        dm = xgb.DMatrix(X.tocsr()[idx])
    else:
        dm = xgb.DMatrix(np.asarray(X)[idx])

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    contrib = np.asarray(booster.predict(dm, pred_contribs=True))  # (n, p+1)
    sv = contrib[:, :-1]
    bias = contrib[:, -1]

    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_signed = np.mean(sv, axis=0)

    names = load_feature_names(dataset_dir)
    if names is None or len(names) != len(mean_abs):
        names = [f"f{i}" for i in range(len(mean_abs))]

    order = np.argsort(-mean_abs)
    if top_features > 0:
        order = order[:top_features]

    out_dir = model_dir / "shap_tables"
    rows = [(names[i], float(mean_abs[i]), float(mean_signed[i])) for i in order]
    write_tsv(out_dir / f"xgb_shap_{split}.tsv", ["feature", "mean_abs_shap", "mean_shap"], rows)
    (out_dir / "meta.json").write_text(json.dumps({"model":"xgb","split":split,"n":int(n),"explain_n":int(len(idx)),
                                                   "bias_mean":float(np.mean(bias))}, indent=2))
    print(f"[DONE] XGB SHAP(contrib) -> {out_dir}")

# ---------- seqCNN IG ----------
def build_onehot(tokens_1d: np.ndarray):
    L = tokens_1d.shape[0]
    oh = np.zeros((4, L), dtype=np.float32)
    for b in range(4):
        oh[b, :] = (tokens_1d == b)
    return oh

def ig_batch(model, x, steps=32):
    import torch
    model.eval()
    baseline = torch.zeros_like(x)
    alphas = torch.linspace(0.0, 1.0, steps=steps, device=x.device).view(steps, 1, 1, 1)
    xs = baseline.unsqueeze(0) + alphas * (x.unsqueeze(0) - baseline.unsqueeze(0))
    xs = xs.requires_grad_(True)
    xs2 = xs.view(-1, *x.shape[1:])
    y = model(xs2).view(-1)
    g = torch.autograd.grad(y.sum(), xs2, create_graph=False)[0]
    g = g.view(steps, x.shape[0], *x.shape[1:]).mean(dim=0)
    ig = (x - baseline) * g
    return ig.detach()

def region_slices(L, head_win, tail_win, mid_bins):
    head = (0, min(head_win, L))
    tail = (max(0, L - tail_win), L)
    mid_start = head[1]
    mid_end = max(mid_start, tail[0])
    mid_len = mid_end - mid_start
    bins = []
    for i in range(mid_bins):
        a = mid_start + int(round(i * mid_len / mid_bins))
        b = mid_start + int(round((i+1) * mid_len / mid_bins))
        bins.append((a, b))
    return [head] + bins + [tail]

def run_seqcnn_ig(dataset_dir: Path, model_dir: Path, split: str, explain_n: int, seed: int,
                 seqbank_dir: Path, head_win: int, tail_win: int, mid_bins: int, ig_steps: int, batch_size: int,
                 project_dir: Path | None = None, weaver_subdir: Path | None = None):
    import torch, joblib

    pt = model_dir / "model.pt"
    if not pt.exists():
        write_skip_meta(model_dir, "seqcnn", split, "missing model.pt (IG needs it)")
        return

    # load seqbank
    meta = json.loads((seqbank_dir / "meta.json").read_text())
    L = int(meta["L"])
    tokens = np.load(seqbank_dir / "seq_tokens.npy", mmap_mode="r")
    id2row = joblib.load(seqbank_dir / "seq_id_to_row.pkl")

    # load split Seq_ID list (dataset_dir first, then weaver fallback)
    try:
        tab, seq_col = find_split_table_with_seqid(dataset_dir, split, project_dir, weaver_subdir)
    except Exception as e:
        write_skip_meta(model_dir, "seqcnn", split, f"cannot locate Seq_ID split table: {e}")
        return

    seq_ids = load_seq_ids_from_table(tab, seq_col)
    n = len(seq_ids)
    if n == 0:
        write_skip_meta(model_dir, "seqcnn", split, f"split table has 0 rows: {tab}")
        return

    idx = sample_indices(n, explain_n, seed)

    # architecture (与你训练脚本要一致；这版仍是“简单两层CNN”)
    class SimpleSeqCNN(torch.nn.Module):
        def __init__(self, L):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(4, 64, kernel_size=7, padding=3)
            self.conv2 = torch.nn.Conv1d(64, 64, kernel_size=7, padding=3)
            self.relu = torch.nn.ReLU()
            self.pool = torch.nn.AdaptiveMaxPool1d(1)
            self.fc = torch.nn.Linear(64, 1)
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x).squeeze(-1)

    model = SimpleSeqCNN(L)
    sd = torch.load(pt, map_location="cpu")
    if isinstance(sd, dict) and any(k.startswith("conv") for k in sd.keys()):
        model.load_state_dict(sd, strict=True)
    elif isinstance(sd, dict) and "state_dict" in sd:
        model.load_state_dict(sd["state_dict"], strict=True)
    else:
        write_skip_meta(model_dir, "seqcnn", split, f"unknown model.pt format: {pt}")
        return

    region_names = ["head"] + [f"mid{i+1}" for i in range(mid_bins)] + ["tail"]
    reg_abs = np.zeros(len(region_names), dtype=np.float64)
    reg_sig = np.zeros(len(region_names), dtype=np.float64)
    base_abs = np.zeros(4, dtype=np.float64)
    base_sig = np.zeros(4, dtype=np.float64)
    slices = region_slices(L, head_win, tail_win, mid_bins)

    ids_sel = [seq_ids[i] for i in idx]
    used = 0

    for st in range(0, len(ids_sel), batch_size):
        batch_ids = ids_sel[st:st+batch_size]
        oh_list = []
        for sid in batch_ids:
            r = id2row.get(sid, None)
            if r is None:
                continue
            oh_list.append(build_onehot(np.asarray(tokens[r], dtype=np.uint8)))
        if len(oh_list) == 0:
            continue

        x = torch.from_numpy(np.stack(oh_list, axis=0))
        ig = ig_batch(model, x, steps=ig_steps).numpy()

        base_abs += np.sum(np.abs(ig), axis=(0,2))
        base_sig += np.sum(ig, axis=(0,2))

        pos = np.sum(ig, axis=1)  # (B,L)
        for ri, (a, b) in enumerate(slices):
            seg = pos[:, a:b]
            reg_abs[ri] += float(np.sum(np.abs(seg)))
            reg_sig[ri] += float(np.sum(seg))

        used += ig.shape[0]

    if used == 0:
        write_skip_meta(model_dir, "seqcnn", split, "IG used=0 (all Seq_ID unmapped?)",
                        {"split_table": str(tab)})
        return

    reg_abs /= used
    reg_sig /= used
    base_abs /= used
    base_sig /= used

    out_dir = model_dir / "attr_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(out_dir / f"ig_region_{split}.tsv", ["region","mean_abs_ig","mean_ig"],
              [(region_names[i], float(reg_abs[i]), float(reg_sig[i])) for i in range(len(region_names))])
    bases = ["A","C","G","T"]
    write_tsv(out_dir / f"ig_base_{split}.tsv", ["base","mean_abs_ig","mean_ig"],
              [(bases[i], float(base_abs[i]), float(base_sig[i])) for i in range(4)])

    (out_dir / "meta.json").write_text(json.dumps({
        "model":"seqcnn","split":split,"n":n,"explain_n":int(used),
        "L":L,"head_win":head_win,"tail_win":tail_win,"mid_bins":mid_bins,
        "ig_steps":ig_steps, "split_table": str(tab)
    }, indent=2))
    print(f"[DONE] seqCNN IG -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model", required=True, choices=["rf","xgb","seqcnn"])
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--explain_n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--top_features", type=int, default=500)

    ap.add_argument("--seqbank_dir", default="")
    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=1)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)

    # fallback source for Seq_ID split tables (DataWeaver split_files)
    ap.add_argument("--project_dir", default="")
    ap.add_argument("--weaver_subdir", default="")

    args = ap.parse_args()
    d = Path(args.dataset_dir)
    m = Path(args.model_dir)

    if args.model == "rf":
        run_rf_shap(d, m, args.split, args.explain_n, args.seed, args.top_features)
    elif args.model == "xgb":
        run_xgb_contrib(d, m, args.split, args.explain_n, args.seed, args.top_features)
    else:
        if not args.seqbank_dir:
            raise RuntimeError("[BAD] seqcnn mode requires --seqbank_dir")
        pdir = Path(args.project_dir) if args.project_dir else None
        wsub = Path(args.weaver_subdir) if args.weaver_subdir else None
        run_seqcnn_ig(d, m, args.split, args.explain_n, args.seed, Path(args.seqbank_dir),
                      args.head_win, args.tail_win, args.mid_bins, args.ig_steps, args.batch_size,
                      project_dir=pdir, weaver_subdir=wsub)

if __name__ == "__main__":
    main()
