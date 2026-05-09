#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

TOK2BASE = {0:"PAD",1:"A",2:"C",3:"G",4:"T",5:"N"}

def load_tokens(ds: Path, split: str):
    p = ds / f"tokens_{split}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    npz = np.load(p)
    return npz["X"]  # uint8 (n,L)

def load_checkpoint(model_dir: Path):
    for name in ["model.pt", "model_best.pt", "checkpoint.pt"]:
        p = model_dir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--explain_n", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--head_win", type=int, default=30)
    ap.add_argument("--tail_win", type=int, default=30)
    ap.add_argument("--mid_bins", type=int, default=3)
    ap.add_argument("--ig_steps", type=int, default=32)
    ap.add_argument("--local_top_pos", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    from captum.attr import LayerIntegratedGradients

    ds = Path(args.dataset_dir)
    md = Path(args.model_dir)
    out = md / "attr_tables_lenaware"
    out.mkdir(parents=True, exist_ok=True)

    ckpt_path = load_checkpoint(md)
    if ckpt_path is None:
        (out / "MISSING_MODEL_PT.txt").write_text(
            "Missing checkpoint (model.pt). Rerun seqcnn training with checkpoint saving enabled.\n",
            encoding="utf-8"
        )
        print(f"[SKIP] missing model.pt -> {md}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    mj = md / "metrics.json"
    if mj.exists():
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
            cfg = {**m.get("config", {}), **cfg}
        except Exception:
            pass

    embed_dim = int(cfg.get("embed_dim", 16))
    dropout = float(cfg.get("dropout", 0.2))
    y_clip = float(cfg.get("y_clip", 6.0))

    def pick_device():
        if args.device == "cpu": return torch.device("cpu")
        if args.device == "cuda": return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = pick_device()
    print(f"[INFO] device={device} ckpt={ckpt_path.name}")

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

    model = SeqCNN(embed_dim=embed_dim, dropout=dropout, y_clip=y_clip).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    Xtok = load_tokens(ds, args.split)  # (n,L)
    L = int(Xtok.shape[1])

    y = np.load(ds / f"y_{args.split}.npy").astype(np.float32, copy=False)
    meta = pd.read_csv(ds / f"meta_{args.split}.tsv.gz", sep="\t", compression="gzip")

    n = Xtok.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(np.arange(n), size=min(args.explain_n, n), replace=False)
    Xsub = Xtok[idx]
    ysub = y[idx]
    metasub = meta.iloc[idx].reset_index(drop=True)
    n_expl = int(len(Xsub))

    head = int(args.head_win)
    tail = int(args.tail_win)
    bins = int(args.mid_bins)

    region_list = ["head30"] + [f"mid{i}" for i in range(1, bins+1)] + ["tail30"]

    # position-based (absolute positions)
    pos_tok_n = np.zeros((L,), dtype=np.int64)
    pos_abs_sum = np.zeros((L,), dtype=np.float64)
    pos_sum = np.zeros((L,), dtype=np.float64)

    # from-end position (0=last base, 1=second last ...)
    from_end_tok_n = np.zeros((tail,), dtype=np.int64)
    from_end_abs_sum = np.zeros((tail,), dtype=np.float64)
    from_end_sum = np.zeros((tail,), dtype=np.float64)

    # region/base
    region_tok_n = {r:0 for r in region_list}
    region_abs_sum = {r:0.0 for r in region_list}
    region_sum = {r:0.0 for r in region_list}

    base_tok_n = {b:0 for b in ["A","C","G","T","N"]}
    base_abs_sum = {b:0.0 for b in ["A","C","G","T","N"]}
    base_sum = {b:0.0 for b in ["A","C","G","T","N"]}

    baseline = torch.zeros((1, L), dtype=torch.long, device=device)
    lig = LayerIntegratedGradients(lambda t: model(t), model.emb)

    bs = int(args.batch_size)
    local_rows = []

    for st in range(0, Xsub.shape[0], bs):
        ed = min(Xsub.shape[0], st+bs)
        tok = torch.from_numpy(Xsub[st:ed].astype(np.int64, copy=False)).to(device)

        attr = lig.attribute(
            inputs=tok,
            baselines=baseline.expand(tok.shape[0], -1),
            n_steps=int(args.ig_steps)
        )
        attr = attr.detach().cpu().numpy().astype(np.float32)   # (B,L,E)
        pos_attr = attr.sum(axis=2)                              # (B,L)
        pos_attr_abs = np.abs(pos_attr)

        tok_np = tok.detach().cpu().numpy().astype(np.int16)
        mask = (tok_np != 0)

        # absolute position aggregation (vectorized)
        pos_tok_n += mask.sum(axis=0).astype(np.int64)
        pos_abs_sum += (pos_attr_abs * mask).sum(axis=0)
        pos_sum += (pos_attr * mask).sum(axis=0)

        # per-sample region assignment (length-aware)
        for i in range(tok_np.shape[0]):
            row_tok = tok_np[i]
            row_mask = mask[i]
            if not row_mask.any():
                continue
            last = int(np.max(np.where(row_mask)[0]))
            seq_len = last + 1

            # local top positions (optional)
            if args.local_top_pos and int(args.local_top_pos) > 0:
                top = int(args.local_top_pos)
                sv = pos_attr[i]
                jj = np.argsort(np.abs(sv))[::-1][:top]
                local_rows.append({
                    "Seq_ID": str(metasub.loc[st+i, "Seq_ID"]) if "Seq_ID" in metasub.columns else str(st+i),
                    "pair_id": str(metasub.loc[st+i, "pair_id"]) if "pair_id" in metasub.columns else "",
                    "y_true": float(ysub[st+i]),
                    "seq_len": int(seq_len),
                    "top_pos": json.dumps([int(x) for x in jj], ensure_ascii=False),
                    "top_attr": json.dumps([float(sv[x]) for x in jj], ensure_ascii=False),
                })

            mid_start = head
            mid_end = max(seq_len - tail, head)
            mid_len = max(0, mid_end - mid_start)

            # iterate only non-pad positions
            nz = np.where(row_mask)[0]
            for p in nz:
                p = int(p)
                v = float(pos_attr[i, p])
                av = float(pos_attr_abs[i, p])

                # base stats
                b = TOK2BASE.get(int(row_tok[p]), "N")
                if b in base_tok_n:
                    base_tok_n[b] += 1
                    base_abs_sum[b] += av
                    base_sum[b] += v

                # region stats (length-aware tail)
                if p < head:
                    reg = "head30"
                elif p >= max(0, seq_len - tail):
                    reg = "tail30"
                    # from-end index (0=last base)
                    d = (seq_len - 1) - p
                    if 0 <= d < tail:
                        from_end_tok_n[d] += 1
                        from_end_abs_sum[d] += av
                        from_end_sum[d] += v
                else:
                    if mid_len <= 0:
                        reg = "mid1"
                    else:
                        rel = p - mid_start
                        bidx = int(rel * bins / mid_len) + 1
                        bidx = max(1, min(bins, bidx))
                        reg = f"mid{bidx}"

                region_tok_n[reg] += 1
                region_abs_sum[reg] += av
                region_sum[reg] += v

    # write tables
    pos_df = pd.DataFrame({
        "pos0": np.arange(L, dtype=int),
        "tok_n": pos_tok_n,
        "tok_frac": pos_tok_n / max(1, n_expl),
        "sum_abs_attr": pos_abs_sum,
        "sum_attr": pos_sum,
        "mean_abs_attr_per_token": np.where(pos_tok_n>0, pos_abs_sum / pos_tok_n, 0.0),
        "mean_attr_per_token": np.where(pos_tok_n>0, pos_sum / pos_tok_n, 0.0),
    })
    pos_df.to_csv(out / f"attr_position_{args.split}.tsv.gz", sep="\t", index=False, compression="gzip")

    fe_df = pd.DataFrame({
        "pos_from_end0": np.arange(tail, dtype=int),
        "tok_n": from_end_tok_n,
        "tok_frac": from_end_tok_n / max(1, n_expl),
        "sum_abs_attr": from_end_abs_sum,
        "sum_attr": from_end_sum,
        "mean_abs_attr_per_token": np.where(from_end_tok_n>0, from_end_abs_sum / from_end_tok_n, 0.0),
        "mean_attr_per_token": np.where(from_end_tok_n>0, from_end_sum / from_end_tok_n, 0.0),
    })
    fe_df.to_csv(out / f"attr_pos_from_end_{args.split}.tsv.gz", sep="\t", index=False, compression="gzip")

    reg_rows = []
    total_tokens = int(sum(region_tok_n.values()))
    for r in region_list:
        tn = int(region_tok_n[r])
        reg_rows.append({
            "region": r,
            "tok_n": tn,
            "tok_frac": tn / max(1, total_tokens),
            "sum_abs_attr": float(region_abs_sum[r]),
            "sum_attr": float(region_sum[r]),
            "mean_abs_attr_per_token": float(region_abs_sum[r] / tn) if tn>0 else 0.0,
            "mean_attr_per_token": float(region_sum[r] / tn) if tn>0 else 0.0,
        })
    pd.DataFrame(reg_rows).sort_values("sum_abs_attr", ascending=False)\
      .to_csv(out / f"attr_region_{args.split}.tsv", sep="\t", index=False)

    base_rows = []
    total_base = int(sum(base_tok_n.values()))
    for b in ["A","C","G","T","N"]:
        tn = int(base_tok_n[b])
        base_rows.append({
            "base": b,
            "tok_n": tn,
            "tok_frac": tn / max(1, total_base),
            "sum_abs_attr": float(base_abs_sum[b]),
            "sum_attr": float(base_sum[b]),
            "mean_abs_attr_per_token": float(base_abs_sum[b] / tn) if tn>0 else 0.0,
            "mean_attr_per_token": float(base_sum[b] / tn) if tn>0 else 0.0,
        })
    pd.DataFrame(base_rows).sort_values("sum_abs_attr", ascending=False)\
      .to_csv(out / f"attr_base_{args.split}.tsv", sep="\t", index=False)

    if local_rows:
        pd.DataFrame(local_rows).to_csv(out / f"attr_local_toppos{int(args.local_top_pos)}_{args.split}.tsv.gz",
                                        sep="\t", index=False, compression="gzip")

    note = [
        f"n_explain={n_expl}",
        f"L_pad={L}",
        f"head_win={head} tail_win={tail} mid_bins={bins}",
        f"total_tokens_nonpad={total_tokens}",
        f"tail_tokens_total={region_tok_n['tail30']}",
        "NOTE: tail/head/mid are computed per-sequence using true length (last non-PAD token).",
    ]
    (out / f"coverage_note_{args.split}.txt").write_text("\n".join(note) + "\n", encoding="utf-8")

    info = {
        "method": "LayerIntegratedGradients(embedding) length-aware tail/head",
        "dataset_dir": str(ds),
        "model_dir": str(md),
        "split": args.split,
        "explain_n": n_expl,
        "ig_steps": int(args.ig_steps),
        "head_win": head,
        "tail_win": tail,
        "mid_bins": bins,
        "seed": int(args.seed),
        "ckpt": str(ckpt_path),
    }
    (out / f"attr_info_{args.split}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n",
                                                      encoding="utf-8")
    print(f"[DONE] seqcnn IG(len-aware) -> {out}")

if __name__ == "__main__":
    main()
