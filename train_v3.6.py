## train_v3.6.py (Paper-Ready Metrics)
import sys
import data_encoder 
import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from datetime import datetime 

# --- 1. 配置参数 ---
DATA_PATH = './processed_data/catalyst_dataset_v3_latest.pt' 
RESULT_DIR = './training_logs_v3_6' # 升级到 v3.6 日志
os.makedirs(RESULT_DIR, exist_ok=True)

# 超参
BATCH_SIZE = 32         
LR = 1e-3               
EPOCHS = 3000           
D_MODEL = 128           
N_HEAD = 4              
NUM_LAYERS = 3          
DROPOUT = 0.1           

PATIENCE = 300          
SCHEDULER_PATIENCE = 100 

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 V3.6 (Paper-Ready) 训练启动 | 设备: {device}")

# --- 2. 核心组件 ---
class CatalystTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=1, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.sequence_embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model*4,
            dropout=DROPOUT, batch_first=True
        )
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.feature_embedding(src).unsqueeze(1)
        tgt = self.sequence_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.output_head(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# --- 3. 🆕 科研指标计算函数 ---
def calculate_metrics(pred, target):
    """
    计算科研常用的评估指标
    :param pred: 预测值 Tensor
    :param target: 真实值 Tensor
    :return: mse, rmse, mae, r2
    """
    # 1. MSE (Loss)
    mse = torch.mean((pred - target) ** 2)
    
    # 2. RMSE
    rmse = torch.sqrt(mse)
    
    # 3. MAE
    mae = torch.mean(torch.abs(pred - target))
    
    # 4. R2 (Coefficient of Determination)
    # R2 = 1 - SS_res / SS_tot
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - pred) ** 2)
    
    # 添加一个小 epsilon 防止除零 (虽然不太可能)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    return mse.item(), rmse.item(), mae.item(), r2.item()

# --- 4. 训练主流程 ---
def train():
    if not os.path.exists(DATA_PATH):
        print(f"❌ 找不到 {DATA_PATH}，请先运行 build_dataset_v3.py")
        return

    data_packet = torch.load(DATA_PATH, weights_only=False)
    X = data_packet['X'].to(device)
    y = data_packet['y'].to(device)
    filenames = data_packet['filenames'] 
    
    # Shuffle
    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]
    filenames = [filenames[i] for i in indices.tolist()]
    
    # 划分测试集 (8个样本)
    test_count = 8
    train_size = len(X) - test_count
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    test_filenames = filenames[train_size:]

    # 🔥 修复：这里之前漏了，现在加回来了！
    times = data_packet.get('times', np.arange(y.shape[1])) 
    
    indices = torch.randperm(len(X))
    X = X[indices]
    y = y[indices]
    filenames = [filenames[i] for i in indices.tolist()]
    
    test_count = 8
    train_size = len(X) - test_count
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    test_filenames = filenames[train_size:]
    
    print(f"📊 数据集概览: 总计 {len(X)} 条 | Train: {len(X_train)} | Test: {len(X_test)}")

    model = CatalystTransformer(input_dim=X.shape[1], d_model=D_MODEL, nhead=N_HEAD, num_layers=NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss() 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE)

    best_test_loss = float('inf')
    best_model_state = None
    best_metrics = {} # 存储最佳时刻的各项指标
    patience_counter = 0

    print(f"\n🔥 开始训练 (监控 R2, RMSE, MAE)...")
    
    tgt_mask_train = generate_square_subsequent_mask(y_train.size(1)).to(device)
    tgt_mask_test = generate_square_subsequent_mask(y_test.size(1)).to(device)

    for epoch in range(EPOCHS):
        # Train
        model.train()
        optimizer.zero_grad()
        output = model(X_train, y_train, tgt_mask=tgt_mask_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test, y_test, tgt_mask=tgt_mask_test)
            # 计算全套指标
            mse_val, rmse_val, mae_val, r2_val = calculate_metrics(test_pred, y_test)
        
        scheduler.step(mse_val)
        
        # 早停逻辑 (以 MSE 为准)
        if mse_val < best_test_loss:
            best_test_loss = mse_val
            best_model_state = copy.deepcopy(model.state_dict())
            best_metrics = {'rmse': rmse_val, 'mae': mae_val, 'r2': r2_val}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch+1) % 100 == 0:
            lr_curr = optimizer.param_groups[0]['lr']
            # 打印更详细的日志
            print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {lr_curr:.1e}")
            print(f"   📉 Train MSE: {loss.item():.6f}")
            print(f"   🧪 Test  MSE: {mse_val:.6f} | RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | R2: {r2_val:.4f}")
        
        #if patience_counter >= PATIENCE:
        #    print(f"\n🛑 早停触发! (Epoch {epoch+1})")
        #    break

    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        print("\n✅ 训练结束，最佳模型指标:")
        print(f"   MSE : {best_test_loss:.6f}")
        print(f"   RMSE: {best_metrics['rmse']:.4f}")
        print(f"   MAE : {best_metrics['mae']:.4f}")
        print(f"   R2  : {best_metrics['r2']:.4f} (越接近1越好)")

    # --- 可视化 (Best vs Worst) ---
    print("\n🧪 生成最终可视化报告...")
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test, y_test, tgt_mask=tgt_mask_test)
        
        # 计算每个样本的 MSE 用于排序
        sample_losses = torch.mean((test_pred - y_test)**2, dim=[1, 2])
        best_idx = torch.argmin(sample_losses).item()
        worst_idx = torch.argmax(sample_losses).item()
        
        plot_configs = [
            (best_idx, "Best Prediction", "green"),
            (worst_idx, "Worst Prediction", "red")
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for idx, title, color in plot_configs:
            plt.figure(figsize=(10, 6))
            
            true_curve = y_test[idx].cpu().numpy().flatten()
            pred_curve = test_pred[idx].cpu().numpy().flatten()
            
            # 物理锁
            pred_curve[0] = 1.0 
            
            # 计算该样本的独立 R2
            _, _, _, r2_single = calculate_metrics(torch.tensor(pred_curve), torch.tensor(true_curve))
            
            sample_name = test_filenames[idx]
            
            plt.plot(times, true_curve, 'b-o', label='Real Data', linewidth=2)
            plt.plot(times, pred_curve, color=color, linestyle='--', marker='x', label='AI Prediction', linewidth=2)
            
            # 标题加入 R2，显得更专业
            plt.title(f'[{title}]\nSource: {sample_name}\nR² = {r2_single:.4f} | MSE = {sample_losses[idx]:.6f}')
            plt.xlabel('Time (min)')
            plt.ylabel('C/C0')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            
            img_path = os.path.join(RESULT_DIR, f'result_v3.6_{title.split()[0].lower()}_{timestamp}.png')
            plt.savefig(img_path)
            print(f"📈 {title} 保存: {img_path}")

    model_path = os.path.join(RESULT_DIR, f'model_v3.6_final_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存: {model_path}")

if __name__ == "__main__":
    train()