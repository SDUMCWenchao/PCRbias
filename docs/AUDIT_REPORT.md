# 脚本审计与开源化整理报告

生成日期：2026-05-09

## 1. 文件概况

- 原始上传文件：`scripts.tar.gz` 与 `chapter23_scripts.zip`
- 整理后仓库草案：`pcr-bias-thesis-pipeline_draft/`
- 文件类型统计：{'.gitattributes': 1, '.gitignore': 1, 'markdown': 1, 'yaml': 2, 'tsv': 3, 'shell': 44, 'python': 148, 'slurm': 5, 'r': 2, 'toml': 1, 'txt': 1}
- 已排除：`*.log`、`__pycache__/` 等运行产物

## 2. 两部分脚本的功能边界

### Chapter 2–3 脚本：`legacy/chapter2_3/`

该部分结构较清晰，整体是从元数据、单端测序数据质控、丰度表构建、预期设计表、序列注释、BLAST 校验、物种标签统一、阈值敏感性分析，到序列特征提取和统计分析的闭环流程。核心可复用入口是：

- `00_setup_dirs.sh`
- `02_single_end_qc_trim_filter.sh`
- `03_build_master_long_abundance.py`
- `10_compute_bias_and_nontarget.py`
- `17_recompute_bias_multi_threshold_from_renorm.py`
- `27_run_chapter3_pipeline_resume.sh`

其中 `27_run_chapter3_pipeline_resume.sh` 比 `26_run_chapter3_pipeline_parallel.sh` 更适合作为公开项目入口，因为它有状态标记、日志目录、RNAfold 路径检测和断点续跑逻辑。

### Chapter 4 脚本：`legacy/chapter4/`

该部分规模更大，包含序列预处理、全局特征与 k-mer 特征计算、PCR/non-PCR 配对统计、机器学习数据集构建、随机森林、XGBoost、1D-CNN、外部数据验证、SHAP/IG 可解释性分析以及论文表格导出。核心逻辑可划分为：

1. 数据准备与 QC：`00_*`、`01_*`
2. 序列特征与 k-mer：`02a_*`–`02e_*`
3. 配对与统计：`03a_*`、`03b_*`、`04a_*`–`04k_*`
4. 模型训练：`06a_*`–`06u_*`
5. 解释性分析：`07*`
6. 论文结果导出：`08*`
7. 外部验证：`ext*`

## 3. 静态检查结果

- Python 语法：第一轮检查未发现编译错误。
- Bash/Slurm 语法：第一轮 `bash -n` 检查未发现语法错误。
- Python 外部依赖初步识别：biopython, captum, joblib, matplotlib, numpy, orjson, pandas, pyfaidx, scikit-learn, scipy, seaborn, shap, statsmodels, torch, xgboost
- 非 Python 命令依赖初步识别：fastp (2), cutadapt (5), blastn (2), makeblastdb (2), RNAfold (7), samtools (7), bwa (4), vsearch (2), qiime (1), seqkit (3), sbatch (34)

## 4. 主要问题

1. **硬编码路径过多**：检测到 `/datapool` 绝对路径 126 处。公开前应统一改为 `--project-dir`、环境变量或 `config.yaml`。
2. **历史版本较多**：存在 `_v2`、`_v3`、`external`、`topbias`、`resplit` 等多套历史版本。公开前应标注 canonical 版本。
3. **依赖未正式锁定**：已生成 `requirements.txt` 和 `environment.yml` 草案，但仍需用实际服务器环境校正版本号。
4. **缺少最小示例数据和测试**：目前只能做语法级 smoke test，无法验证科学输出。
5. **Slurm 脚本仍绑定本机路径**：多数 `#SBATCH -o/-e`、`PROJECT`、`PROJECT_DIR` 写死服务器目录。
6. **许可证未确定**：若目标是开源项目，必须加入真实开源许可证。

## 5. 已完成的整理

- 将第2–3章脚本放入 `legacy/chapter2_3/`
- 将第4章脚本放入 `legacy/chapter4/`
- 删除日志文件和 Python 缓存
- 生成 `.gitignore`，避免 FASTQ/BAM/模型/日志等大文件进入 Git
- 生成 `environment.yml` 与 `requirements.txt` 草案
- 生成 `configs/config.example.yaml`
- 生成 `tools/smoke_test.sh`
- 生成 `tools/audit_hardcoded_paths.py`
- 生成 `docs/SCRIPT_INVENTORY.tsv`
- 生成 GitHub 部署说明

## 6. 后续重构优先级

### P0：公开前必须处理

- 删除或忽略所有真实数据、日志、模型文件、绝对路径、私有账号路径
- 确定许可证
- 写清楚入口脚本、输入文件格式和输出目录
- 对 Chapter 2–3 使用 `27_run_chapter3_pipeline_resume.sh` 作为主入口

### P1：提高可复现性

- 所有脚本统一支持 `--config configs/config.yaml` 或 `--project-dir`
- 将 Slurm 脚本参数化
- 增加 toy dataset 和 smoke test
- 固定关键依赖版本

### P2：正式软件化

- 抽取公共函数到 `src/pcr_bias/`
- 增加命令行入口，例如 `pcrbias chapter23 run`
- 用 Snakemake/Nextflow 或统一 Bash runner 管理全流程
- 增加 GitHub Actions 做语法检查和最小示例运行


## 8. 2026-05-09 许可证与 GitHub 仓库信息更新

- 目标仓库已确定为 `https://github.com/SDUMCWenchao/PCR_bias`。
- 许可证已更新为 AGPL-3.0，并加入根目录 `LICENSE`。
- 已加入 `CITATION.cff`、GitHub Actions、Issue 模板、PR 模板、`Makefile`、公开发布检查清单和重构路线图。
- 硬编码路径仍作为 legacy provenance 问题保留，CI 使用 `--legacy-mode report` 报告但不失败。
