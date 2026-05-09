library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# === 配置路径 ===
# 如果在服务器运行，请修改为服务器绝对路径
# 如果在本地运行，请修改为本地路径
input_dir <- "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/analysis/09_Feature_Summary"
output_dir <- "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/analysis/09_Feature_Summary/plots"

if (!dir.exists(output_dir)) dir.create(output_dir)

# === 1. 读取数据 ===
message("Reading Feature Table...")
df_feat <- read.delim(file.path(input_dir, "ALL_UNIQUE_FEATURES.tsv"), stringsAsFactors = FALSE)

message("Reading Enrichment Viz Table...")
df_viz <- read.delim(file.path(input_dir, "ALL_ENRICHMENT_VIZ.tsv"), stringsAsFactors = FALSE)

# === 2. Plot A: 物理特征分布 (Violin + Boxplot) ===
# 准备数据: 将不同指标标准化以便画在一起，或者分开画
# 这里我们重点关注 MFE, Entropy, Runs

p1 <- ggplot(df_feat, aes(x = "", y = MFE)) + 
  geom_violin(fill = "lightblue", alpha = 0.5) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(title = "Distribution of Minimum Free Energy (MFE)", 
       y = "MFE (kcal/mol)", x = "") +
  theme_bw()

p2 <- ggplot(df_feat, aes(x = "", y = Entropy)) + 
  geom_violin(fill = "lightgreen", alpha = 0.5) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(title = "Distribution of Shannon Entropy", 
       y = "Entropy (bits)", x = "") +
  theme_bw()

p3 <- ggplot(df_feat, aes(x = "", y = Runs)) + 
  geom_violin(fill = "pink", alpha = 0.5) +
  geom_boxplot(width = 0.2, outlier.size = 0.5) +
  labs(title = "Distribution of Runs (Sequence Complexity)", 
       y = "Number of Runs", x = "") +
  theme_bw()

# 组合图
pdf(file.path(output_dir, "01_Feature_Distributions.pdf"), width = 10, height = 4)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

# === 3. Plot B: 结构 vs 复杂度散点图 (Density Scatter) ===
# 使用 Hexbin 或 Density 避免点过多重叠
p_scatter <- ggplot(df_feat, aes(x = Entropy, y = MFE)) +
  stat_binhex(bins = 50) + # 六边形热图展示密度
  scale_fill_gradient(low = "lightgray", high = "darkblue", name = "Count") +
  geom_smooth(method = "lm", color = "red", linetype = "dashed", size = 0.5) +
  labs(title = "Correlation: Structural Stability vs Sequence Complexity",
       subtitle = "Lower MFE = More Stable Structure; Lower Entropy = Less Complex",
       x = "Shannon Entropy",
       y = "Minimum Free Energy (MFE)") +
  theme_bw()

ggsave(file.path(output_dir, "02_Structure_vs_Complexity.pdf"), p_scatter, width = 8, height = 6)

# === 4. Plot C: 碱基富集区分布 (Enrichment Zones) ===
# 我们想看不同 Motif 的富集区通常出现在序列的什么位置
# 选取几个关键 Motif: GC, A, T (代表 Poly-A/T)

target_motifs <- c("GC", "A", "T", "G", "C")
df_viz_sub <- df_viz %>% filter(Motif %in% target_motifs & Threshold == 0.8)

# 计算每个位置的覆盖密度
p_enrich <- ggplot(df_viz_sub) +
  # 使用线段表示每个富集区
  geom_segment(aes(x = Start, xend = End, y = Motif, yend = Motif, color = Motif), 
               alpha = 0.05, size = 2) +
  labs(title = "Positional Distribution of High-Density Motifs (Threshold > 0.8)",
       subtitle = "Where do extreme sequence compositions occur?",
       x = "Position in Sequence (bp)",
       y = "Motif") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(file.path(output_dir, "03_Enrichment_Zone_Map.pdf"), p_enrich, width = 10, height = 6)

message("All plots generated in: ", output_dir)