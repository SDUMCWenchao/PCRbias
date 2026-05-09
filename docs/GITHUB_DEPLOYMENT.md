# GitHub deployment guide

Target repository:

```text
https://github.com/SDUMCWenchao/PCR_bias
```

License:

```text
AGPL-3.0
```

## Upload from a local computer or server

```bash
unzip PCR_bias_open_source_v3.zip
cd PCR_bias

git init
git add .
git commit -m "Initial public release draft"
git branch -M main
git remote add origin https://github.com/SDUMCWenchao/PCR_bias.git
git push -u origin main
```

## After upload

Run these checks locally and through GitHub Actions:

```bash
bash tools/smoke_test.sh
python tools/audit_hardcoded_paths.py --legacy-mode report
```

## Important exclusion rules

Do not commit:

- raw FASTQ/FQ files
- BAM/SAM/CRAM files
- trained model binaries
- Slurm log files
- private `configs/config.yaml`
- credentials or server-specific secrets
- unpublished private datasets
