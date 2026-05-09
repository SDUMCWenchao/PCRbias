#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------- metrics ----------------
def r2_score(y, p):
    y = np.asarray(y); p = np.asarray(p)
    ssr = np.sum((y - p) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ssr / (sst + 1e-12))

def rmse(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.sqrt(np.mean((y - p) ** 2)))

def mae(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean(np.abs(y - p)))

def sign_acc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return float(np.mean((y >= 0) == (p >= 0)))

def spearman(y, p):
    rr = spearmanr(y, p)
    return float(rr.correlation) if np.isfinite(rr.correlation) else float("nan")

def pair_spearman(meta_df, y, p, min_pair_n=30):
    if "pair_id" not in meta_df.columns:
        return float("nan"), 0
    out = []
    for pid, g in meta_df.groupby("pair_id"):
        idx = g.index.values
        if len(idx) < min_pair_n:
            continue
        rr = spearmanr(y[idx], p[idx])
        if np.isfinite(rr.correlation):
            out.append(float(rr.correlation))
    if not out:
        return float("nan"), 0
    return float(np.mean(out)), int(len(out))

# ---------------- seq fetch & encode ----------------
def try_pyfaidx(fasta_path: Path):
    try:
        from pyfaidx import Fasta
        fa = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True)
        return fa
    except Exception:
        return None

def fetch_seqs(fasta_path: Path, ids, use_pyfaidx=True):
    ids = [str(x) for x in ids]
    got = {}
    fa = try_pyfaidx(fasta_path) if use_pyfaidx else None
    if fa is not None:
        miss = 0
        for sid in ids:
            try:
                got[sid] = str(fa[sid])
            except Exception:
                miss += 1
        if miss:
            print(f"[WARN] pyfaidx missing {miss} seqs (will be PAD only)")
        return got

    wanted = set(ids)
    cur = None
    buf = []
    keep = False
    with fasta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None and keep:
                    got[cur] = "".join(buf).upper()
                cur = line[1:].strip().split()[0]
                buf = []
                keep = cur in wanted
            else:
                if keep:
                    buf.append(line.strip())
        if cur is not None and keep:
            got[cur] = "".join(buf).upper()
    miss = len(wanted) - len(got)
    if miss:
        print(f"[WARN] scan missing {miss} seqs (will be PAD only)")
    return got

# 0:PAD, 1:A,2:C,3:G,4:T,5:N/other
MAP = np.zeros(256, dtype=np.uint8) + 5
MAP[ord('A')] = 1
MAP[ord('C')] = 2
MAP[ord('G')] = 3
MAP[ord('T')] = 4
MAP[ord('N')] = 5

def encode_batch_to_tokens(seqs, max_len):
    n = len(seqs)
    X = np.zeros((n, max_len), dtype=np.uint8)
    for i, s in enumerate(seqs):
        if not s:
            continue
        b = s.upper().encode("ascii", errors="ignore")
        L = min(len(b), max_len)
        if L <= 0:
            continue
        X[i, :L] = MAP[np.frombuffer(b[:L], dtype=np.uint8)]
    return X

def maybe_build_cache(dataset_dir: Path, fasta_path: Path, max_len: int, use_pyfaidx=True, force=False):
    for split in ["train", "val", "test"]:
        out_npz = dataset_dir / f"tokens_{split}.npz"
        if out_npz.exists() and not force:
            continue
        meta = pd.read_csv(dataset_dir / f"meta_{split}.tsv.gz", sep="\t", compression="gzip")
        ids = meta["Seq_ID"].astype(str).tolist()
        seq_map = fetch_seqs(fasta_path, ids, use_pyfaidx=use_pyfaidx)
        seqs = [seq_map.get(sid, "") for sid in ids]
        X = encode_batch_to_tokens(seqs, max_len)
        np.savez_compressed(out_npz, X=X)
        print(f"[DONE] cached {out_npz} shape={X.shape}")

def load_tokens(dataset_dir: Path, split: str):
    npz = np.load(dataset_dir / f"tokens_{split}.npz")
    return npz["X"]

# ---------------- model ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fasta", default="/datapool/zhangw/duwenchao/var/2511_PCR_Bias/analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta")
    ap.add_argument("--max_len", type=int, default=500)
    ap.add_argument("--y_clip", type=float, default=6.0)
    ap.add_argument("--use_pyfaidx", action="store_true")
    ap.add_argument("--force_cache", action="store_true")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--embed_dim", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--min_pair_n", type=int, default=10)

    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    d = Path(args.dataset_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fasta_path = Path(args.fasta)
    maybe_build_cache(d, fasta_path, args.max_len, use_pyfaidx=args.use_pyfaidx, force=args.force_cache)

    Xtr = load_tokens(d, "train")
    Xva = load_tokens(d, "val")
    Xte = load_tokens(d, "test")

    ytr = np.load(d / "y_train.npy").astype(np.float32, copy=False)
    yva = np.load(d / "y_val.npy").astype(np.float32, copy=False)
    yte = np.load(d / "y_test.npy").astype(np.float32, copy=False)

    wtr = np.load(d / "w_train.npy").astype(np.float32, copy=False)
    wva = np.load(d / "w_val.npy").astype(np.float32, copy=False)
    wte = np.load(d / "w_test.npy").astype(np.float32, copy=False)

    meta_va = pd.read_csv(d / "meta_val.tsv.gz", sep="\t", compression="gzip")
    meta_te = pd.read_csv(d / "meta_test.tsv.gz", sep="\t", compression="gzip")

    def pick_device():
        if args.device == "cpu":
            return torch.device("cpu")
        if args.device == "cuda":
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = pick_device()
    print(f"[INFO] device={device}")

    class TokDS(Dataset):
        def __init__(self, X, y, w):
            self.X = X
            self.y = y
            self.w = w
        def __len__(self): return len(self.y)
        def __getitem__(self, i):
            return self.X[i], self.y[i], self.w[i]

    tr_ld = DataLoader(TokDS(Xtr, ytr, wtr), batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_ld = DataLoader(TokDS(Xva, yva, wva), batch_size=args.batch_size, shuffle=False, num_workers=0)
    te_ld = DataLoader(TokDS(Xte, yte, wte), batch_size=args.batch_size, shuffle=False, num_workers=0)

    class SeqCNN(nn.Module):
        def __init__(self, vocab=6, embed_dim=16, dropout=0.2, y_clip=6.0):
            super().__init__()
            self.y_clip = float(y_clip)
            self.emb = nn.Embedding(vocab, embed_dim, padding_idx=0)
            self.conv = nn.Sequential(
                nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(128, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
        def forward(self, tok):
            x = self.emb(tok.long())      # (B,L,E)
            x = x.transpose(1,2)          # (B,E,L)
            x = self.conv(x)
            x = self.pool(x)              # (B,128,1)
            y = self.head(x).squeeze(1)   # (B,)
            y = torch.tanh(y) * self.y_clip
            return y

    model = SeqCNN(embed_dim=args.embed_dim, dropout=args.dropout, y_clip=args.y_clip).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def weighted_huber(pred, y, w, delta=1.0):
        w = w / (w.mean() + 1e-12)
        err = pred - y
        abs_err = torch.abs(err)
        quad = torch.minimum(abs_err, torch.tensor(delta, device=abs_err.device))
        lin  = abs_err - quad
        loss = 0.5 * quad * quad + delta * lin
        return torch.mean(loss * w)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best = float("inf")
    best_state = None
    bad = 0

    def eval_split(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb, wb in loader:
                xb = xb.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    pred = model(xb)
                ps.append(pred.detach().cpu().numpy())
                ys.append(yb.numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        return rmse(y, p), y, p

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb, wb in tr_ld:
            xb = xb.to(device)
            yb = yb.to(device)
            wb = wb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                pred = model(xb)
                loss = weighted_huber(pred, yb, wb, delta=1.0)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

        cur, _, _ = eval_split(va_ld)
        print(f"[EPOCH] {ep:03d} val_rmse={cur:.4f}")

        if cur + 1e-6 < best:
            best = cur
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("[INFO] early stop")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval
    _, yv, pv = eval_split(va_ld)
    _, yt, pt = eval_split(te_ld)

    vps, vpn = pair_spearman(meta_va, yv, pv, min_pair_n=args.min_pair_n)
    tps, tpn = pair_spearman(meta_te, yt, pt, min_pair_n=args.min_pair_n)

    metrics = {
        "val": {"r2": r2_score(yv, pv), "rmse": rmse(yv, pv), "mae": mae(yv, pv),
                "spearman": spearman(yv, pv), "sign_acc": sign_acc(yv, pv), "n": int(len(yv))},
        "test": {"r2": r2_score(yt, pt), "rmse": rmse(yt, pt), "mae": mae(yt, pt),
                 "spearman": spearman(yt, pt), "sign_acc": sign_acc(yt, pt), "n": int(len(yt))},
        "val_pair": {"pair_spearman_mean": vps, "pair_spearman_n": vpn, "min_pair_n": args.min_pair_n},
        "test_pair": {"pair_spearman_mean": tps, "pair_spearman_n": tpn, "min_pair_n": args.min_pair_n},
        "config": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # --- NEW: always save checkpoint for attribution ---
    import torch
    torch.save({"state_dict": model.state_dict(), "config": vars(args)}, out / "model.pt")

    # Save preds
    for split, meta, y, p in [("val", meta_va, yv, pv), ("test", meta_te, yt, pt)]:
        df = meta.copy()
        df["y_true"] = y
        df["y_pred"] = p
        df.to_csv(out / f"pred_{split}.tsv.gz", sep="\t", index=False, compression="gzip")

    print(f"[DONE] seqcnn -> {out}  test_r2={metrics['test']['r2']:.4f}  test_spear={metrics['test']['spearman']:.4f}")

if __name__ == "__main__":
    main()
