import os
import subprocess
import glob

# === 配置路径 ===
base_dir = "/path/to/PCR_bias_chapter4/"
raw_data_dir = os.path.join(base_dir, "raw_data")
output_dir = os.path.join(base_dir, "analysis", "01_QC")

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# === 获取所有 .fq 文件 ===
# 注意：如果文件名是 .fastq 或 .fq.gz，请根据实际情况修改后缀
fq_files = glob.glob(os.path.join(raw_data_dir, "*.fq")) + glob.glob(os.path.join(raw_data_dir, "*.fastq"))

print(f"Found {len(fq_files)} fastq files.")

# === 运行 FastQC ===
for fq in fq_files:
    cmd = f"fastqc -o {output_dir} -t 4 {fq}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

# === 运行 MultiQC ===
print("Running MultiQC to aggregate results...")
multiqc_cmd = f"multiqc {output_dir} -o {output_dir}"
subprocess.run(multiqc_cmd, shell=True)

print(f"QC analysis complete. Please check report at: {output_dir}/multiqc_report.html")