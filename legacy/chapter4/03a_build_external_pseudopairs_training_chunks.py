#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, hashlib, math, re
from pathlib import Path
from collections import defaultdict

IUPAC_OK = set(list("ACGTNRYKMSWBDHV"))
HEX32 = re.compile(r"^[0-9a-fA-F]{32}$")

def log(msg: str):
    print(msg, flush=True)

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def clean_token(s: str) -> str:
    x = (s or "").strip()
    for _ in range(2):
        if len(x) >= 2 and ((x[0] == '"' and x[-1] == '"') or (x[0] == "'" and x[-1] == "'")):
            x = x[1:-1].strip()
    if x.startswith(">"):
        x = x[1:].strip()
    return x

def looks_like_seq(s: str) -> bool:
    if len(s) < 15:
        return False
    ss = clean_token(s).upper().replace("U","T").replace(" ","")
    return all(c in IUPAC_OK for c in ss)

def fasta_iter(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    header = None
    parts = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts)
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
        if header is not None:
            yield header, "".join(parts)

def sniff_abundance_format(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip("\n").split(",")
    lower = [h.strip().lower() for h in header]
    has_exp = any(x in lower for x in ["experiment","exp","sample","condition"])
    has_seq = any(x in lower for x in ["sequence","seq","seq_id","scaffold","id"])
    has_val = any(x in lower for x in ["abundance","count","reads","value"])
    if has_exp and has_seq and has_val:
        return "long"
    return "wide"

def parse_cycle(exp_name: str):
    m = re.search(r"(?:^|[^A-Za-z0-9])PCR(\d+)(?:$|[^A-Za-z0-9])", exp_name, flags=re.I)
    if not m:
        m = re.search(r"PCR(\d+)", exp_name, flags=re.I)
    return int(m.group(1)) if m else None

def choose_control_and_yes(experiments: list[str], control_exp: str | None):
    exp_cycles = [(e, parse_cycle(e)) for e in experiments]
    pcr_like = [e for e,c in exp_cycles if c is not None]
    use_exps = pcr_like if pcr_like else experiments

    if control_exp:
        if control_exp not in use_exps:
            raise ValueError(f"[BAD] control_exp={control_exp} not in {use_exps}")
        ctrl = control_exp
    else:
        cyc2 = [(e,c) for e,c in [(e, parse_cycle(e)) for e in use_exps] if c is not None]
        ctrl = sorted(cyc2, key=lambda x: x[1])[0][0] if cyc2 else sorted(use_exps)[0]

    yes = [e for e in use_exps if e != ctrl]
    yes_cyc = [(e, parse_cycle(e)) for e in yes]
    if any(c is not None for _,c in yes_cyc):
        yes = [e for e,_ in sorted(yes_cyc, key=lambda x: (x[1] is None, x[1] if x[1] is not None else 10**9, x[0]))]
    else:
        yes = sorted(yes)
    return ctrl, yes

def split_of_seq(seq_id: str, seed: int, frac_train: float, frac_val: float):
    h = hashlib.md5(f"{seed}|{seq_id}".encode("utf-8")).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    if u < frac_train:
        return "train"
    if u < frac_train + frac_val:
        return "val"
    return "test"

def support_ok(c_yes: float, c_no: float, min_support: float, mode: str):
    if min_support <= 0:
        return True
    if mode == "both":
        return (c_yes >= min_support) and (c_no >= min_support)
    if mode == "any":
        return (c_yes >= min_support) or (c_no >= min_support)
    if mode == "sum":
        return (c_yes + c_no) >= min_support
    raise ValueError(mode)

# --------- mapping: external abundance row-id -> external_test Seq_ID ----------

def build_external_seq_index(external_fa: Path):
    header_set = set()
    md5_to_seqid = {}
    n = 0
    for hid, seq in fasta_iter(external_fa):
        n += 1
        header_set.add(hid)
        s = clean_token(seq).upper().replace("U","T").replace(" ","")
        md5_to_seqid[md5_hex(s)] = hid
    log(f"[INFO] external fasta index: n={n} headers={len(header_set)} md5={len(md5_to_seqid)} from {external_fa}")
    return header_set, md5_to_seqid

def list_mapping_candidates(ds_dir: Path, abundance_name: str):
    cands = []
    for pat in ["*.fa", "*.fna", "*.fasta", "*.fa.gz", "*.fna.gz", "*.fasta.gz"]:
        cands += list(ds_dir.glob(pat))
    for pat in ["*.tsv", "*.tsv.gz", "*.csv", "*.csv.gz"]:
        cands += list(ds_dir.glob(pat))
    cands = [p for p in cands if p.name != abundance_name]
    def score(p: Path):
        nm = p.name.lower()
        s = 0
        for k in ["map","seq","sequence","oligo","id","uniq","fasta","fa"]:
            if k in nm: s += 2
        if "abundance" in nm: s -= 5
        return s
    return sorted(set(cands), key=score, reverse=True)

def sample_raw_ids_from_abundance(csv_path: Path, n=2000):
    fmt = sniff_abundance_format(csv_path)
    ids = []
    if fmt == "wide":
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rr = csv.reader(f)
            header = next(rr, None)
            if not header:
                return fmt, [], []
            exps = [h.strip() for h in header[1:]]
            for row in rr:
                if not row: 
                    continue
                ids.append(clean_token(row[0]))
                if len(ids) >= n:
                    break
            return fmt, exps, ids
    # long
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f)
        cols = {c.lower(): c for c in (dr.fieldnames or [])}
        def pick(cands):
            for c in cands:
                if c in cols: return cols[c]
            return None
        col_seq = pick(["sequence","seq","seq_id","scaffold","id"])
        col_exp = pick(["experiment","exp","sample","condition"])
        exps = set()
        for row in dr:
            if col_exp:
                exps.add((row.get(col_exp) or "").strip())
            if col_seq:
                ids.append(clean_token(row.get(col_seq) or ""))
            if len(ids) >= n:
                break
    return fmt, sorted([e for e in exps if e]), ids

def try_build_idmap_from_fasta(fa_path: Path, md5_to_seqid: dict[str,str], limit=None):
    idmap = {}
    n = 0
    hit = 0
    for hid, seq in fasta_iter(fa_path):
        n += 1
        raw_id = clean_token(hid)
        s = clean_token(seq).upper().replace("U","T").replace(" ","")
        sid = md5_to_seqid.get(md5_hex(s), "")
        if sid:
            idmap[raw_id] = sid
            hit += 1
        if limit and n >= limit:
            break
    return idmap, n, hit

def sniff_delim_and_header(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        first = f.readline()
    # simple sniff
    if "\t" in first and first.count("\t") >= first.count(","):
        return "\t"
    return ","

def try_build_idmap_from_table(tab_path: Path, md5_to_seqid: dict[str,str], limit=200000):
    """
    Expect columns include id + sequence; auto-detect by content.
    """
    delim = sniff_delim_and_header(tab_path)
    opener = gzip.open if str(tab_path).endswith(".gz") else open
    idmap = {}
    n = 0
    hit = 0
    with opener(tab_path, "rt", encoding="utf-8", errors="replace", newline="") as f:
        rr = csv.reader(f, delimiter=delim)
        header = next(rr, None)
        if not header:
            return idmap, n, hit
        # find seq col by checking first few rows
        # if header has obvious names, use them
        lower = [h.strip().lower() for h in header]
        seq_idx = None
        id_idx = None
        for i, h in enumerate(lower):
            if h in ("sequence","seq","oligo","dna","seqstr"):
                seq_idx = i
            if h in ("id","seq_id","name","label","key"):
                id_idx = i
        # fallback: assume 2-col table
        if seq_idx is None and len(header) == 2:
            id_idx, seq_idx = 0, 1
        if id_idx is None and len(header) >= 2:
            id_idx = 0
        if seq_idx is None and len(header) >= 2:
            seq_idx = 1

        for row in rr:
            if not row or max(id_idx, seq_idx) >= len(row):
                continue
            rid = clean_token(row[id_idx])
            s = clean_token(row[seq_idx])
            if not rid or not s:
                continue
            if not looks_like_seq(s):
                continue
            seq = s.upper().replace("U","T").replace(" ","")
            sid = md5_to_seqid.get(md5_hex(seq), "")
            if sid:
                idmap[rid] = sid
                hit += 1
            n += 1
            if n >= limit:
                break
    return idmap, n, hit

def build_dataset_idmap(ds_dir: Path, abundance_name: str, header_set: set[str], md5_to_seqid: dict[str,str], sample_ids: list[str]):
    """
    Return: idmap dict raw_id -> Seq_ID, and a log string.
    """
    # if sample ids already are headers, no need
    direct = sum(1 for x in sample_ids if x in header_set)
    if direct > 0:
        return {}, f"direct_header_hits={direct}/{len(sample_ids)} (no extra idmap)"

    cands = list_mapping_candidates(ds_dir, abundance_name)[:30]
    best = (0.0, None, None, None)  # frac, path, idmap, info
    sample_set = set([x for x in sample_ids if x])

    for p in cands:
        idmap = {}
        info = ""
        try:
            if any(p.name.endswith(ext) for ext in [".fa",".fna",".fasta",".fa.gz",".fna.gz",".fasta.gz"]):
                idmap, n, hit = try_build_idmap_from_fasta(p, md5_to_seqid)
                info = f"FASTA nrec={n} hit2external={hit} idmap={len(idmap)}"
            elif any(p.name.endswith(ext) for ext in [".tsv",".tsv.gz",".csv",".csv.gz"]):
                idmap, n, hit = try_build_idmap_from_table(p, md5_to_seqid)
                info = f"TABLE nrow={n} hit2external={hit} idmap={len(idmap)}"
            else:
                continue
        except Exception:
            continue

        if not idmap:
            continue
        overlap = sum(1 for x in sample_set if x in idmap)
        frac = overlap / max(1, len(sample_set))
        if frac > best[0]:
            best = (frac, p, idmap, info)

    if best[1] is None:
        return {}, "idmap=NONE (no mapping file matched sample ids)"
    return best[2], f"idmap={best[1].name} overlap={best[0]:.3f} {best[3]}"

def normalize_row_id(raw: str, header_set: set[str], md5_to_seqid: dict[str,str], idmap: dict[str,str]):
    x = clean_token(raw)
    if not x:
        return ""
    if x in header_set:
        return x
    if x in idmap:
        return idmap[x]
    if looks_like_seq(x):
        seq = x.upper().replace("U","T").replace(" ","")
        return md5_to_seqid.get(md5_hex(seq), "")
    if HEX32.match(x) and x in header_set:
        return x
    return ""

def read_abundance(csv_path: Path, header_set: set[str], md5_to_seqid: dict[str,str], idmap: dict[str,str]):
    fmt = sniff_abundance_format(csv_path)
    exp2seq2val = defaultdict(dict)

    if fmt == "wide":
        with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            rr = csv.reader(f)
            header = next(rr, None)
            if not header:
                return fmt, [], exp2seq2val, {"drop":0,"keep":0}
            exps = [h.strip() for h in header[1:]]
            drop = keep = 0
            for row in rr:
                if not row: 
                    continue
                sid = normalize_row_id(row[0], header_set, md5_to_seqid, idmap)
                if not sid:
                    drop += 1
                    continue
                for j, exp in enumerate(exps, start=1):
                    if j >= len(row): 
                        continue
                    s = (row[j] or "").strip()
                    if s == "" or s == "0":
                        continue
                    try:
                        v = float(s)
                    except Exception:
                        continue
                    if v == 0:
                        continue
                    exp2seq2val[exp][sid] = exp2seq2val[exp].get(sid, 0.0) + v
                keep += 1
            return fmt, exps, exp2seq2val, {"drop":drop,"keep":keep}

    # long
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        dr = csv.DictReader(f)
        cols = {c.lower(): c for c in (dr.fieldnames or [])}
        def pick(cands):
            for c in cands:
                if c in cols: return cols[c]
            return None
        col_seq = pick(["sequence","seq","seq_id","scaffold","id"])
        col_exp = pick(["experiment","exp","sample","condition"])
        col_val = pick(["abundance","count","reads","value"])
        if not (col_seq and col_exp and col_val):
            raise ValueError(f"[BAD] long format but cannot find columns in {csv_path}")
        exps = set()
        drop = keep = 0
        for row in dr:
            raw = (row.get(col_seq) or "")
            exp = (row.get(col_exp) or "").strip()
            if exp:
                exps.add(exp)
            sid = normalize_row_id(raw, header_set, md5_to_seqid, idmap)
            if not sid:
                drop += 1
                continue
            vraw = (row.get(col_val) or "").strip()
            try:
                v = float(vraw)
            except Exception:
                continue
            if v == 0:
                continue
            exp2seq2val[exp][sid] = exp2seq2val[exp].get(sid, 0.0) + v
            keep += 1
    return fmt, sorted([e for e in exps if e]), exp2seq2val, {"drop":drop,"keep":keep}

# --------- features / write chunks ----------

def load_needed_features(core_dir: Path, needed: set[str]):
    feats = {}
    core_files = sorted(core_dir.glob("*.core.tsv.gz"))
    if not core_files:
        raise FileNotFoundError(f"[BAD] no core chunks in {core_dir}")
    all_cols = None
    hit = 0
    for fp in core_files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            dr = csv.DictReader(f, delimiter="\t")
            if all_cols is None:
                all_cols = list(dr.fieldnames or [])
            for row in dr:
                sid = row.get("Seq_ID", "")
                if sid in needed:
                    feats[sid] = row
                    hit += 1
    if not feats:
        raise RuntimeError("[BAD] loaded 0 features: still mismatch after idmap. This means your dataset sequence space != external_test fasta.")
    feat_cols = ["Seq_ID"] + sorted([c for c in (all_cols or []) if c != "Seq_ID"])
    log(f"[INFO] loaded features: hits={hit}/{len(needed)}")
    return feats, feat_cols

def write_training_chunks(out_dir: Path, rows_iter, fieldnames: list[str], rows_per_chunk: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = out_dir / "index.tsv"
    chunk_i = 0
    cur_n = 0
    cur_fp = None
    cur_fh = None
    written = []
    n_total = 0

    def open_new():
        nonlocal chunk_i, cur_n, cur_fp, cur_fh
        if cur_fh:
            cur_fh.close()
        cur_fp = out_dir / f"chunk_{chunk_i:04d}.train.tsv.gz"
        cur_fh = gzip.open(cur_fp, "wt", encoding="utf-8")
        cur_fh.write("\t".join(fieldnames) + "\n")
        cur_n = 0
        chunk_i += 1

    open_new()
    for r in rows_iter:
        if cur_n >= rows_per_chunk:
            written.append((cur_fp.name, cur_n))
            open_new()
        cur_fh.write("\t".join(str(r.get(k,"")) for k in fieldnames) + "\n")
        cur_n += 1
        n_total += 1
    if cur_fh:
        cur_fh.close()
    if cur_fp:
        written.append((cur_fp.name, cur_n))

    with idx.open("w", encoding="utf-8") as f:
        f.write("chunk\trows\n")
        for name, n in written:
            f.write(f"{name}\t{n}\n")
    log(f"[DONE] training_chunks={len(written)} rows={n_total} -> {out_dir}")
    log(f"[DONE] index -> {idx}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", required=True)
    ap.add_argument("--pcrbias_root", required=True)
    ap.add_argument("--dataset_glob", default="analysis/data/external_datasets/*")
    ap.add_argument("--abundance_name", default="abundance_by_experiment.csv")
    ap.add_argument("--control_exp", default=None)

    ap.add_argument("--min_support", type=float, default=2.0)
    ap.add_argument("--min_support_mode", choices=["both","any","sum"], default="both")
    ap.add_argument("--eps", type=float, default=1e-12)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--split_fracs", default="0.8,0.1,0.1")
    ap.add_argument("--rows_per_chunk", type=int, default=200000)

    ap.add_argument("--out_subdir", default="analysis_results/03_DataWeaver")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample_n", type=int, default=2000, help="sample ids for idmap scoring")
    args = ap.parse_args()

    project = Path(args.project_dir).resolve()
    pcrbias = Path(args.pcrbias_root).resolve()
    out_root = project / args.out_subdir
    out_chunks = out_root / "training_chunks"
    out_meta = out_root / "pairs_summary.tsv"

    if out_root.exists() and not args.force:
        raise SystemExit(f"[ABORT] {out_root} exists. Use --force to overwrite.")

    frac_train, frac_val, frac_test = [float(x) for x in args.split_fracs.split(",")]
    if abs(frac_train + frac_val + frac_test - 1.0) > 1e-6:
        raise ValueError("split_fracs must sum to 1.0")

    external_fa = project / "analysis_results/01_Sequences/ALL_UNIQUE_SEQUENCES.fasta"
    if not external_fa.exists():
        gz = external_fa.with_suffix(".fasta.gz")
        if gz.exists():
            external_fa = gz
        else:
            raise FileNotFoundError(f"[BAD] missing {external_fa} (or .gz)")

    header_set, md5_to_seqid = build_external_seq_index(external_fa)
    core_dir = project / "analysis_results/02_Features/core_chunks"

    patterns = [p.strip() for p in args.dataset_glob.split(",") if p.strip()]
    ds_set = set()
    for pat in patterns:
        for d in pcrbias.glob(pat):
            if d.is_dir() and (d / args.abundance_name).exists():
                ds_set.add(d)
    ds_dirs = sorted(ds_set)

    if not ds_dirs:
        raise SystemExit("[ABORT] no datasets found")
    log(f"[INFO] datasets found = {len(ds_dirs)}")

    ds_info = []
    needed = set()

    for d in ds_dirs:
        ab = d / args.abundance_name
        fmt, exps_hdr, sample_ids = sample_raw_ids_from_abundance(ab, n=args.sample_n)
        idmap, idmap_info = build_dataset_idmap(d, args.abundance_name, header_set, md5_to_seqid, sample_ids)
        fmt2, exps, exp2seq2val, st = read_abundance(ab, header_set, md5_to_seqid, idmap)
        # keep experiments list (wide uses header even if some exp empty)
        exps_use = exps if exps else exps_hdr

        log(f"[INFO] dataset={d.name} fmt={fmt2} exps={len(exps_use)} {idmap_info} keep_rows={st['keep']} drop_rows={st['drop']}")

        if not exps_use:
            log(f"[WARN] dataset={d.name} exps=0, skip")
            continue

        ctrl, yes_list = choose_control_and_yes(exps_use, args.control_exp)
        pairs = [(ctrl, y) for y in yes_list]

        for ctrl_exp, yes_exp in pairs:
            m_no = exp2seq2val.get(ctrl_exp, {})
            m_yes = exp2seq2val.get(yes_exp, {})
            for sid in (set(m_no.keys()) | set(m_yes.keys())):
                c_no = float(m_no.get(sid, 0.0))
                c_yes = float(m_yes.get(sid, 0.0))
                if support_ok(c_yes, c_no, args.min_support, args.min_support_mode):
                    needed.add(sid)

        ds_info.append((d, exp2seq2val, pairs))
        log(f"[INFO] dataset={d.name} ctrl={ctrl} yes={len(yes_list)}")

    if not needed:
        raise SystemExit("[ABORT] needed_ids=0. This means after idmap+mapping, no rows survived. Check audit output; likely missing the ID->sequence file in each dataset directory.")

    log(f"[INFO] needed Seq_ID total = {len(needed)}")

    feats_map, feat_cols = load_needed_features(core_dir, needed)
    feature_cols = [c for c in feat_cols if c != "Seq_ID"]

    out_root.mkdir(parents=True, exist_ok=True)
    out_chunks.mkdir(parents=True, exist_ok=True)

    base_cols = [
        "dataset", "exp_no", "exp_yes", "pcr_cycle_no", "pcr_cycle_yes",
        "compare_id", "split",
        "pair_id", "yes_file_id", "no_file_id",
        "Seq_ID",
        "count_yes", "count_no", "total_yes", "total_no",
        "rel_yes", "rel_no", "log2fc",
    ]
    fieldnames = base_cols + feature_cols

    def totals_for(exp2seq2val):
        return {exp: float(sum(m.values())) for exp, m in exp2seq2val.items()}

    pairs_summary = []
    pair_counter = 0

    def row_iter():
        nonlocal pair_counter
        for d, exp2seq2val, pairs in ds_info:
            ds = d.name
            tots = totals_for(exp2seq2val)
            for no_exp, yes_exp in pairs:
                m_no = exp2seq2val.get(no_exp, {})
                m_yes = exp2seq2val.get(yes_exp, {})
                t_no = float(tots.get(no_exp, 0.0))
                t_yes = float(tots.get(yes_exp, 0.0))
                if t_no <= 0 or t_yes <= 0:
                    continue
                cyc_no = parse_cycle(no_exp)
                cyc_yes = parse_cycle(yes_exp)
                cid = f"{ds}__{yes_exp}_vs_{no_exp}"
                n_rows = 0
                for sid in (set(m_no.keys()) | set(m_yes.keys())):
                    c_no = float(m_no.get(sid, 0.0))
                    c_yes = float(m_yes.get(sid, 0.0))
                    if not support_ok(c_yes, c_no, args.min_support, args.min_support_mode):
                        continue
                    feat = feats_map.get(sid)
                    if feat is None:
                        continue
                    rel_no = c_no / t_no
                    rel_yes = c_yes / t_yes
                    log2fc = math.log((rel_yes + args.eps) / (rel_no + args.eps), 2)
                    sp = split_of_seq(sid, args.seed, frac_train, frac_val)
                    pair_counter += 1
                    out = {
                        "dataset": ds,
                        "exp_no": no_exp,
                        "exp_yes": yes_exp,
                        "pcr_cycle_no": cyc_no if cyc_no is not None else "",
                        "pcr_cycle_yes": cyc_yes if cyc_yes is not None else "",
                        "compare_id": cid,
                        "split": sp,
                        "pair_id": f"{cid}__{pair_counter}",
                        "yes_file_id": yes_exp,
                        "no_file_id": no_exp,
                        "Seq_ID": sid,
                        "count_yes": c_yes,
                        "count_no": c_no,
                        "total_yes": t_yes,
                        "total_no": t_no,
                        "rel_yes": rel_yes,
                        "rel_no": rel_no,
                        "log2fc": log2fc,
                    }
                    for c in feature_cols:
                        out[c] = feat.get(c, "")
                    n_rows += 1
                    yield out
                pairs_summary.append({
                    "dataset": ds, "exp_no": no_exp, "exp_yes": yes_exp,
                    "pcr_cycle_no": cyc_no if cyc_no is not None else "",
                    "pcr_cycle_yes": cyc_yes if cyc_yes is not None else "",
                    "rows": n_rows, "total_no": t_no, "total_yes": t_yes,
                    "min_support": args.min_support, "min_support_mode": args.min_support_mode
                })
                log(f"[DONE] {cid}: rows={n_rows}")

    write_training_chunks(out_chunks, row_iter(), fieldnames, args.rows_per_chunk)

    if pairs_summary:
        with out_meta.open("w", encoding="utf-8", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=list(pairs_summary[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(pairs_summary)
        log(f"[DONE] pairs summary -> {out_meta}")

if __name__ == "__main__":
    main()
