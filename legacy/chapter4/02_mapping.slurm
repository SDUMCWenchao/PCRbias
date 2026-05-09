#!/bin/bash
#SBATCH --job-name=Map_PCRBias
#SBATCH --output=logs/%x_%A_%a.out   # 输出日志: JobName_JobID_ArrayID.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # 每个样本分配 4 个核心
#SBATCH --mem=4G                     # 每个样本分配 4G 内存 (线粒体数据足够)
#SBATCH --array=2-23%16              # 处理第2到23行，同时最多运行16个任务 (16*4=64核)

# === 1. 定义路径 ===
BASE_DIR="/datapool/zhangw/duwenchao/var/2511_PCR_Bias"
RAW_DIR="${BASE_DIR}/raw_data"
REF_DIR="${BASE_DIR}/refs"
OUT_DIR="${BASE_DIR}/analysis/02_Mapping"
META_FILE="${BASE_DIR}/samples_meta.tsv"

# 创建输出目录
mkdir -p ${OUT_DIR}
mkdir -p logs

# === 2. 解析 Metadata (根据 Array ID 读取对应行) ===
# 获取 TSV 文件的第 N 行 (N = SLURM_ARRAY_TASK_ID)
# samples_meta.tsv  结构: file_id(1), sample_name(2), species(3), ...
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${META_FILE})

# 提取关键信息
FILE_ID=$(echo "$LINE" | awk '{print $1}')      # e.g., H12
SAMPLE_NAME=$(echo "$LINE" | awk '{print $2}')  # e.g., MP10_12
SPECIES=$(echo "$LINE" | awk '{print $3}')      # e.g., 10mix, donkey, cattle

echo "Processing Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Sample: ${FILE_ID} (${SAMPLE_NAME})"
echo "Species: ${SPECIES}"

# === 3. 自动选择参考基因组 ===
# 根据 species 列决定使用哪个 fasta
if [[ "$SPECIES" == "10mix" ]]; then
    REF_FASTA="${REF_DIR}/combined_mito_ref.fasta"
elif [[ "$SPECIES" == "donkey" ]]; then
    REF_FASTA="${REF_DIR}/donkey.fasta"
elif [[ "$SPECIES" == "cattle" ]]; then
    REF_FASTA="${REF_DIR}/cattle.fasta"
elif [[ "$SPECIES" == "pig" ]]; then
    REF_FASTA="${REF_DIR}/pig.fasta"
else
    echo "Error: Unknown species '$SPECIES' for sample $FILE_ID"
    exit 1
fi

echo "Selected Reference: ${REF_FASTA}"

# === 4. 确定输入文件 ===
# 假设原始数据文件名格式为 "ID.fq" 或 "ID.fastq"
# 先尝试 FILE_ID (H12.fq)
FQ_FILE="${RAW_DIR}/${FILE_ID}.fq"

if [ ! -f "$FQ_FILE" ]; then
    # 如果找不到，尝试 .fastq
    FQ_FILE="${RAW_DIR}/${FILE_ID}.fastq"
fi

# 如果还找不到，尝试用 SAMPLE_NAME (MP10_12.fq) 
if [ ! -f "$FQ_FILE" ]; then
    FQ_FILE="${RAW_DIR}/${SAMPLE_NAME}.fq"
fi

if [ ! -f "$FQ_FILE" ]; then
    echo "Error: Input file not found for ID ${FILE_ID} or ${SAMPLE_NAME}"
    exit 1
fi

echo "Input FastQ: ${FQ_FILE}"

# === 5. 运行 Mapping (BWA + Samtools) ===
BAM_OUT="${OUT_DIR}/${FILE_ID}.sorted.bam"

# RG Header: 增加 Read Group 信息，这对后续 GATK/FreeBayes 等分析很重要
RG_TAG="@RG\tID:${FILE_ID}\tSM:${SAMPLE_NAME}\tPL:ILLUMINA\tLB:${SPECIES}_${FILE_ID}"

cmd="bwa mem -t ${SLURM_CPUS_PER_TASK} -R '${RG_TAG}' ${REF_FASTA} ${FQ_FILE} | \
     samtools view -Sb - | \
     samtools sort -@ 2 -o ${BAM_OUT} -"

echo "Running: bwa mem ..."
eval $cmd

# === 6. 建立索引和简单统计 ===
samtools index ${BAM_OUT}
samtools flagstat ${BAM_OUT} > ${BAM_OUT}.flagstat

echo "Done. Output: ${BAM_OUT}"