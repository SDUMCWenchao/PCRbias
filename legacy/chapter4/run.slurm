#!/bin/bash
#SBATCH --job-name=ml
#SBATCH --output=otu_ml.log
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G


cd /path/to/PCR_bias_chapter4

python scripts/06a_prepare_ml_datasets.py \
  --max_train 2000000 --max_val 250000 --max_test 250000 \
  --mid_bins 3 --kmer_x_mode presence
