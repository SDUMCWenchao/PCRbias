import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# === 配置 ===
BASE_DIR = "/path/to/PCR_bias_chapter4"
INPUT_DIR = os.path.join(BASE_DIR, "analysis/10_Modeling_Data_Deep")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis/17_CNN_Results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设备配置 (优先使用 GPU，否则 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 1. 数据预处理工具 ===
def seq_to_onehot(seq, max_len):
    """
    将 DNA 序列转换为 One-Hot 矩阵 (4 x L)
    A=[1,0,0,0], C=[0,1,0,0], G=[0,0,1,0], T=[0,0,0,1]
    不足 max_len 的补 0
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    onehot = np.zeros((4, max_len), dtype=np.float32)
    
    for i, base in enumerate(seq):
        if i >= max_len: break
        if base in mapping:
            onehot[mapping[base], i] = 1.0
            
    return onehot

class PCRDataset(Dataset):
    def __init__(self, sequences, targets, max_len):
        self.sequences = sequences
        self.targets = targets
        self.max_len = max_len
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        y = self.targets[idx]
        x = seq_to_onehot(seq, self.max_len)
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)

# === 2. CNN 模型定义 ===
class PCR_CNN(nn.Module):
    def __init__(self, seq_len):
        super(PCR_CNN, self).__init__()
        
        # Conv1: 捕捉短 Motif (k=3-5)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv2: 捕捉更长的模式组合
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Conv3: 高层特征
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveMaxPool1d(1) # Global Max Pooling
        
        # Fully Connected
        self.fc1 = nn.Linear(256, 128)
        self.fc_relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1) # 回归输出
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(self.fc_relu1(self.fc1(x)))
        x = self.fc2(x)
        return x.squeeze()

# === 3. 训练函数 ===
def train_cnn(file_path):
    pair_name = os.path.basename(file_path).replace("_Deep.csv", "")
    print(f"\nTraining CNN for: {pair_name}...")
    
    # 读取数据
    try:
        df = pd.read_csv(file_path)
    except:
        return None
        
    # 清洗
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Bias_Score_Y'], inplace=True)
    
    seqs = df['Sequence'].tolist()
    targets = df['Bias_Score_Y'].values
    
    if len(seqs) < 100:
        print("  Skipping: Not enough data.")
        return None
        
    # 确定最大长度 (取 95% 分位，避免极长序列浪费显存)
    lens = [len(s) for s in seqs]
    max_len = int(np.percentile(lens, 95))
    # print(f"  Max Seq Len: {max_len}")
    
    # 划分
    X_train, X_test, y_train, y_test = train_test_split(seqs, targets, test_size=0.2, random_state=42)
    
    # DataLoader
    train_ds = PCRDataset(X_train, y_train, max_len)
    test_ds = PCRDataset(X_test, y_test, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 初始化模型
    model = PCR_CNN(max_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环 (Early Stopping 简化版)
    epochs = 30
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 简单验证
        # (这里略去验证集评估以节省代码长度，直接看测试集结果)
        
    # 最终评估
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.numpy())
            
    r2 = r2_score(all_targets, all_preds)
    pearson = pearsonr(all_targets, all_preds)[0]
    
    print(f"  [CNN Result] R2: {r2:.4f}, Pearson: {pearson:.4f}")
    
    return {
        'Pair': pair_name,
        'R2_CNN': r2,
        'Pearson_CNN': pearson
    }

# === 主循环 ===
files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_Deep.csv")))
results = []

# 只跑 Top 样本 (数据量大，跑 CNN 才有意义)
target_files = [f for f in files if "top0.5pct" in f or "top1pct" in f]

for f in target_files:
    res = train_cnn(f)
    if res: results.append(res)

if results:
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "CNN_Performance_Summary.csv"), index=False)
    print("\nCNN Analysis Complete.")