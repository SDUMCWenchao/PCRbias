import os
import subprocess
import glob

# === 1. 配置路径 (请根据实际情况修改) ===
base_dir = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/"
raw_data_dir = os.path.join(base_dir, "raw_data")
# 输出目录
map_dir = os.path.join(base_dir, "analysis", "02_Mapping")
log_dir = os.path.join(map_dir, "logs")

# === 参考基因组设置 (CRITICAL) ===
# 假设你有一个包含所有物种线粒体序列的 fasta 文件
# 如果针对不同样本用不同参考，逻辑会更复杂。这里假设是用一个 Combined Reference 进行竞争性比对 (推荐用于 10mix)
ref_fasta = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/refs/combined_mito_ref.fasta" 

# 检查目录
for d in [map_dir, log_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# === 2. 建立索引 (如果尚未建立) ===
if not os.path.exists(ref_fasta + ".bwt"):
    print(f"Building BWA index for {ref_fasta}...")
    subprocess.run(f"bwa index {ref_fasta}", shell=True)
else:
    print("BWA index found, skipping indexing.")

# === 3. 批量比对 ===
# 获取所有 fq 文件
fq_files = glob.glob(os.path.join(raw_data_dir, "*.fq")) + glob.glob(os.path.join(raw_data_dir, "*.fastq"))

for fq in fq_files:
    sample_name = os.path.basename(fq).split('.')[0]
    bam_out = os.path.join(map_dir, f"{sample_name}.sorted.bam")
    
    # 构建 BWA MEM 命令 (增加 Read Group 信息这对后续分析很重要)
    # 假设是单端测序 (Single End)
    rg_tag = f"@RG\\tID:{sample_name}\\tSM:{sample_name}\\tPL:ILLUMINA"
    
    cmd = (
        f"bwa mem -t 8 -R '{rg_tag}' {ref_fasta} {fq} | "
        f"samtools view -Sb - | "
        f"samtools sort - -o {bam_out}"
    )
    
    # 索引 BAM 文件
    index_cmd = f"samtools index {bam_out}"

    print(f"Mapping {sample_name}...")
    # 运行 Mapping
    with open(os.path.join(log_dir, f"{sample_name}.log"), "w") as logfile:
        subprocess.run(cmd, shell=True, stderr=logfile)
        subprocess.run(index_cmd, shell=True, stderr=logfile)

print("Mapping complete. BAM files are ready for coverage analysis.")