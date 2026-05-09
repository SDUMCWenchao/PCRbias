#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, gzip, math, sqlite3
from pathlib import Path

def pearson(W, xw, x2w, xyw, ysum, y2sum):
    if W <= 0:
        return 0.0
    ex = xw / W
    ex2 = x2w / W
    ey = ysum / W
    ey2 = y2sum / W
    exy = xyw / W
    vx = ex2 - ex*ex
    vy = ey2 - ey*ey
    if vx <= 0 or vy <= 0:
        return 0.0
    cov = exy - ex*ey
    return cov / math.sqrt(vx*vy)

def mean_var(den, s1, s2):
    if den <= 0:
        return (float("nan"), float("nan"))
    m = s1 / den
    v = s2 / den - m*m
    if v < 0: v = 0.0
    return m, v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", default="/path/to/PCR_bias_chapter4")
    ap.add_argument("--cleanup_db", action="store_true")
    args = ap.parse_args()

    project = Path(args.project_dir)
    base_dir = project / "analysis_results" / "04_Stats_Kmer" / "bases"
    part_dir = project / "analysis_results" / "04_Stats_Kmer" / "partials"
    out_dir  = project / "analysis_results" / "04_Stats_Kmer"
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path = out_dir / "kmer_reduce.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-500000;")  # ~500MB

    conn.execute("""
    CREATE TABLE base(
      pair_id TEXT PRIMARY KEY,
      W REAL, ysum REAL, y2sum REAL,
      cy REAL, cn REAL, nrows INTEGER
    );
    """)
    conn.execute("""
    CREATE TABLE feat(
      pair_id TEXT,
      feature TEXT,
      xw REAL, x2w REAL, xyw REAL,
      xcy REAL, x2cy REAL,
      xcn REAL, x2cn REAL,
      PRIMARY KEY(pair_id, feature)
    );
    """)
    conn.commit()

    # reduce base
    base_files = sorted(base_dir.glob("chunk_*.base.tsv.gz"))
    if not base_files:
        raise FileNotFoundError(f"no base files in {base_dir}")

    for fp in base_files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            rows = []
            for row in r:
                rows.append((
                    row["pair_id"],
                    float(row["W"]), float(row["y_sum"]), float(row["y2_sum"]),
                    float(row["cy_total"]), float(row["cn_total"]), int(row["n_rows"])
                ))
        # upsert-add
        conn.executemany("""
        INSERT INTO base(pair_id,W,ysum,y2sum,cy,cn,nrows)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(pair_id) DO UPDATE SET
          W = W + excluded.W,
          ysum = ysum + excluded.ysum,
          y2sum = y2sum + excluded.y2sum,
          cy = cy + excluded.cy,
          cn = cn + excluded.cn,
          nrows = nrows + excluded.nrows;
        """, rows)
        conn.commit()

    # reduce features
    part_files = sorted(part_dir.glob("chunk_*.kmer.partial.tsv.gz"))
    if not part_files:
        raise FileNotFoundError(f"no partial files in {part_dir}")

    for fp in part_files:
        with gzip.open(fp, "rt", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            rows = []
            for row in r:
                rows.append((
                    row["pair_id"], row["feature"],
                    float(row["xw"]), float(row["x2w"]), float(row["xyw"]),
                    float(row["xcy"]), float(row["x2cy"]),
                    float(row["xcn"]), float(row["x2cn"])
                ))
        conn.executemany("""
        INSERT INTO feat(pair_id,feature,xw,x2w,xyw,xcy,x2cy,xcn,x2cn)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(pair_id,feature) DO UPDATE SET
          xw = xw + excluded.xw,
          x2w = x2w + excluded.x2w,
          xyw = xyw + excluded.xyw,
          xcy = xcy + excluded.xcy,
          x2cy = x2cy + excluded.x2cy,
          xcn = xcn + excluded.xcn,
          x2cn = x2cn + excluded.x2cn;
        """, rows)
        conn.commit()

    # write outputs
    corr_out  = out_dir / "pair_kmer_corr.tsv.gz"
    shift_out = out_dir / "pair_kmer_shift.tsv.gz"

    # load base to memory (pairs count很小)
    base = {}
    for pid, W, ysum, y2sum, cy, cn, nrows in conn.execute("SELECT pair_id,W,ysum,y2sum,cy,cn,nrows FROM base"):
        base[pid] = (W, ysum, y2sum, cy, cn, nrows)

    with gzip.open(corr_out, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","feature","n_points","w_sum","pearson_corr"])
        for pid, feat, xw, x2w, xyw in conn.execute("SELECT pair_id,feature,xw,x2w,xyw FROM feat"):
            W, ysum, y2sum, cy, cn, nrows = base[pid]
            c = pearson(W, xw, x2w, xyw, ysum, y2sum)
            w.writerow([pid, feat, nrows, f"{W:.12g}", f"{c:.12g}"])

    with gzip.open(shift_out, "wt", encoding="utf-8") as fo:
        w = csv.writer(fo, delimiter="\t")
        w.writerow(["pair_id","feature","cy_total","cn_total","mean_yes","mean_no","delta_yes_minus_no","sd_yes","sd_no","effect_d"])
        for pid, feat, xcy, x2cy, xcn, x2cn in conn.execute("SELECT pair_id,feature,xcy,x2cy,xcn,x2cn FROM feat"):
            W, ysum, y2sum, cy, cn, nrows = base[pid]
            my, vy = mean_var(cy, xcy, x2cy)
            mn, vn = mean_var(cn, xcn, x2cn)
            sy = math.sqrt(vy) if vy==vy else float("nan")
            sn = math.sqrt(vn) if vn==vn else float("nan")
            delta = (my - mn) if (my==my and mn==mn) else float("nan")
            pooled = math.sqrt((vy+vn)/2.0) if (vy==vy and vn==vn) else float("nan")
            d = (delta/pooled) if (pooled==pooled and pooled>0 and delta==delta) else float("nan")
            w.writerow([pid, feat, f"{cy:.12g}", f"{cn:.12g}",
                        f"{my:.12g}" if my==my else "NA",
                        f"{mn:.12g}" if mn==mn else "NA",
                        f"{delta:.12g}" if delta==delta else "NA",
                        f"{sy:.12g}" if sy==sy else "NA",
                        f"{sn:.12g}" if sn==sn else "NA",
                        f"{d:.12g}" if d==d else "NA"])

    conn.close()
    if args.cleanup_db:
        db_path.unlink(missing_ok=True)

    print(f"[DONE] {corr_out}")
    print(f"[DONE] {shift_out}")

if __name__ == "__main__":
    main()
