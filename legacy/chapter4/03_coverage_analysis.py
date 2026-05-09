import os
import subprocess
import numpy as np
import pandas as pd
import re
import io
import glob

# === 配置路径 ===
BASE_DIR = "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/"
MAPPING_DIR = os.path.join(BASE_DIR, "analysis", "02_Mapping")
META_FILE = os.path.join(BASE_DIR, "samples_meta.tsv")
OUTPUT_FILE = os.path.join(BASE_DIR, "analysis", "03_Bias_Metrics_Summary.tsv")

# === 辅助函数：运行 samtools depth 并计算指标 ===
def calculate_coverage_metrics(bam_path):
    # 使用 samtools depth 获取所有覆盖度 > 0 的位点数据
    cmd = f"samtools depth {bam_path}"
    
    # 执行命令并捕获输出
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error processing {bam_path}: {e}")
        return None

    # 从输出中提取深度值
    depths = []
    # 使用io.StringIO将字符串输出视为文件
    output_stream = io.StringIO(result.stdout)
    
    # 逐行读取samtools depth的输出 (Contig \t Position \t Depth)
    for line in output_stream:
        try:
            # 尝试获取第3列的值 (Depth)
            depth_val = int(line.strip().split('\t')[2])
            depths.append(depth_val)
        except (ValueError, IndexError):
            # 忽略格式错误的行
            continue

    if not depths:
        return {'Mean Coverage': 0, 'Std Dev': 0, 'CV': np.nan, 'Covered Bases (1x)': 0, 'Total Bases': 0}

    # 转换为 NumPy 数组进行计算
    depths_array = np.array(depths)
    
    # 计算核心指标
    mean_cov = np.mean(depths_array)
    std_dev = np.std(depths_array)
    cv = std_dev / mean_cov if mean_cov > 0 else np.nan
    
    # 统计覆盖度 > 1X 的位点数（即有 reads 的位点数）
    covered_bases = len(depths_array)
    total_bases = 0 
    
    # 为了得到Total Bases，我们需要BAM文件的参考序列长度。
    # 这里我们简化处理：假设所有被比对到的位点都是有效参考长度。
    # 实际应用中，应使用samtools view -H | grep @SQ 获取精确的参考长度
    
    # 临时 Total Bases: 我们可以从 samtools flagstat 的输出中估算，
    # 或者从 metadata 中获取参考序列长度。为了简洁，我们只报告 covered_bases。
    
    return {
        'Mean Coverage': round(mean_cov, 2),
        'Std Dev': round(std_dev, 2),
        'CV (PCR Bias Metric)': round(cv, 4),
        'Covered Bases (1x)': covered_bases
    }


# === 3. 主流程 ===
def main_analysis():
    # 找到所有 BAM 文件
    bam_files = glob.glob(os.path.join(MAPPING_DIR, "*.sorted.bam"))
    
    if not bam_files:
        print(f"Error: No BAM files found in {MAPPING_DIR}")
        return

    # 读取 metadata
    meta_df = pd.read_csv(META_FILE, sep='\t')
    
    results = []
    
    for bam_path in bam_files:
        file_id = os.path.basename(bam_path).split('.')[0]
        
        # 查找 metadata
        meta_info = meta_df[meta_df['file_id'] == file_id]
        if meta_info.empty:
            print(f"Warning: Metadata not found for {file_id}. Skipping.")
            continue
        
        # 提取关键元数据
        species = meta_info['species'].iloc[0]
        pcr = meta_info['pcr'].iloc[0]
        locus = meta_info['locus'].iloc[0]
        
        # 计算覆盖度指标
        metrics = calculate_coverage_metrics(bam_path)
        
        if metrics:
            result_row = {
                'file_id': file_id,
                'species': species,
                'locus': locus,
                'pcr': pcr,
                **metrics
            }
            results.append(result_row)

    # 整合结果并输出
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['species', 'pcr', 'locus'], inplace=True)
    results_df.to_csv(OUTPUT_FILE, sep='\t', index=False)
    
    print("\n" + "="*50)
    print(f"Quantitative PCR Bias Analysis Complete.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print("="*50)
    
    # 打印关键对比结果
    print("\n--- Key Metrics (CV: PCR Bias) ---")
    print(results_df[['file_id', 'species', 'locus', 'pcr', 'Mean Coverage', 'CV (PCR Bias Metric)']])


if __name__ == "__main__":
    main_analysis()