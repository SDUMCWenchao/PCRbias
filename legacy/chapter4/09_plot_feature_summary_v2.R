library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(hexbin) # 确保已安装

# === 配置路径 ===
input_dir <- "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/analysis/09_Feature_Summary"
output_dir <- "/datapool/zhangw/duwenchao/var/2511_PCR_Bias/analysis/09_Feature_Summary/plots_v2"

if (!dir.exists(output_dir)) dir.create(output_dir)

# === 1. 读取数据 ===
message("Reading Feature Table...")
df_feat <- read.delim(file.path(input_dir, "ALL_UNIQUE_FEATURES.tsv"), stringsAsFactors = FALSE)

message("Reading Enrichment Viz Table...")
df_viz <- read.delim(file.path(input_dir, "ALL_ENRICHMENT_VIZ.tsv"), stringsAsFactors = FALSE)

# 过滤异常值 (例如 MFE > 0 的通常是无效预测)
df_feat <- df_feat %>% filter(MFE <= 0)

# === 2. Plot A: 物理特征分布 (修复：使用 PNG + 调整边距) ===
p1 <- ggplot(df_feat, aes(x = "All Seqs", y = MFE)) + 
  geom_violin(fill = "#69b3a2", alpha = 0.6) +
  geom_boxplot(width = 0.1, outlier.shape = NA) +
  labs(title = "Minimum Free Energy", y = "MFE (kcal/mol)", x = "") +
  theme_bw() + theme(plot.margin = unit(c(1,1,1,1), "cm"))

p2 <- ggplot(df_feat, aes(x = "All Seqs", y = Entropy)) + 
  geom_violin(fill = "#404080", alpha = 0.6) +
  geom_boxplot(width = 0.1, outlier.shape = NA) +
  labs(title = "Shannon Entropy", y = "Entropy (bits)", x = "") +
  theme_bw() + theme(plot.margin = unit(c(1,1,1,1), "cm"))

p3 <- ggplot(df_feat, aes(x = "All Seqs", y = Runs)) + 
  geom_violin(fill = "#E69F00", alpha = 0.6) +
  geom_boxplot(width = 0.1, outlier.shape = NA) +
  labs(title = "Runs (Complexity)", y = "Run Count", x = "") +
  theme_bw() + theme(plot.margin = unit(c(1,1,1,1), "cm"))

# 保存为 PNG (宽屏，确保显示全)
png(file.path(output_dir, "01_Feature_Distributions.png"), width = 2400, height = 800, res = 150)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

# === 3. Plot B: 结构 vs 复杂度 (修复：使用 Bin2D 热图) ===
p_scatter <- ggplot(df_feat, aes(x = Entropy, y = MFE)) +
  # 使用 bin2d 避免点重叠，使用对数刻度颜色展示密度差异
  stat_bin2d(bins = 100) +
  scale_fill_viridis_c(trans = "log10", option = "C", name = "Seq Count") +
  geom_smooth(method = "lm", color = "white", linetype = "dashed", size = 0.5) +
  labs(title = "Correlation: Structure Stability vs Complexity",
       subtitle = "Density Heatmap (Log Scale)",
       x = "Shannon Entropy (bits)",
       y = "Minimum Free Energy (kcal/mol)") +
  theme_bw()

ggsave(file.path(output_dir, "02_Structure_vs_Complexity.png"), p_scatter, width = 8, height = 6, dpi = 300)

# === 4. Plot C: 碱基富集区分布 (核心修复：改用密度曲线) ===
# 筛选主要 Motif，且阈值较高的区域
target_motifs <- c("GC", "AT", "A", "G", "C")
df_viz_sub <- df_viz %>% 
  filter(Motif %in% target_motifs & Threshold >= 0.7)

if(nrow(df_viz_sub) > 0) {
  # 计算每个富集区的中心点，用于画密度图
  df_viz_sub$Midpoint <- (df_viz_sub$Start + df_viz_sub$End) / 2
  
  p_enrich <- ggplot(df_viz_sub, aes(x = Midpoint, color = Motif, fill = Motif)) +
    # 使用密度曲线展示分布趋势，而不是画几百万条线
    geom_density(alpha = 0.3, size = 1) +
    facet_wrap(~Motif, ncol = 1, scales = "free_y") +
    labs(title = "Positional Density of High-Enrichment Zones (Threshold > 0.7)",
         subtitle = "Where do these motifs typically appear along the sequence?",
         x = "Position in Sequence (bp)",
         y = "Density") +
    theme_bw() +
    theme(strip.background = element_rect(fill = "gray95"))
  
  ggsave(file.path(output_dir, "03_Enrichment_Zone_Density.png"), p_enrich, width = 10, height = 12, dpi = 300)
} else {
  message("Warning: No enrichment zones found with Threshold >= 0.7")
}

message("All V2 plots (PNG format) generated in: ", output_dir)