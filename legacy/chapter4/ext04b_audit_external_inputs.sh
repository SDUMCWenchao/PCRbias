#!/usr/bin/env bash
set -euo pipefail
ROOT="/path/to/PCR_bias_chapter4/external_test/analysis_results/05_ModelInputs_external_topbias_resplit_v1_resplit_v1"

for tag in top1p top0p5p top0p1p; do
  echo "==== ${tag} ===="
  find "${ROOT}/${tag}" -maxdepth 3 -type d -name "no_kmer" | wc -l | awk '{print "no_kmer datasets:",$1}'
  find "${ROOT}/${tag}" -maxdepth 3 -type d -name "kmer_only" | wc -l | awk '{print "kmer_only datasets:",$1}'
  find "${ROOT}/${tag}" -maxdepth 3 -type d -name "all" | wc -l | awk '{print "all datasets:",$1}'

  # spot check: 是否都有 X/y
  bad=0
  while read -r d; do
    for f in X_train.npz X_val.npz X_test.npz y_train.npy y_val.npy y_test.npy feature_names.tsv; do
      [[ -s "${d}/${f}" ]] || bad=$((bad+1))
    done
  done < <(find "${ROOT}/${tag}" -maxdepth 3 -type d -name "no_kmer")
  echo "missing core files count: ${bad}"
done
