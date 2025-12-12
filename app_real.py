## application.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import os
import data_encoder 

# ==========================================
# 1. 配置路径
# ==========================================
MODEL_PATH = './training_logs_v3_6/model_v3.6_final_20251208_164608.pth' 
DATA_PATH = './processed_data/catalyst_dataset_v3_latest.pt'             

# ==========================================
# 2. 智能设备选择逻辑
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# 获取当前设备
device = get_device()

# ==========================================
# 3. 模型结构 (必须与训练代码一致)
# ==========================================
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

class CatalystTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=1, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.feature_embedding = nn.Linear(input_dim, d_model)
        self.sequence_embedding = nn.Linear(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        # 推理时 dropout 设为 0
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
            num_decoder_layers=num_layers, dim_feedforward=d_model*4,
            dropout=0.0, batch_first=True
        )
        self.output_head = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        src = self.feature_embedding(src).unsqueeze(1)
        tgt = self.sequence_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        return self.output_head(output)

# ==========================================
# 4. 资源加载 (修复版：Flatten + Safe Load)
# ==========================================
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
        return None, None, None
    
    # 1. 加载数据集包 (获取 Processor 和 Times)
    # 🔥 修复: weights_only=False 解决安全报错
    checkpoint = torch.load(DATA_PATH, map_location=device, weights_only=False)
    processor = checkpoint['processor']
    
    # 2. 探测输入维度
    dummy_input = {'Catalyst_Type': 'Fe-SAC', 'pH': 7}
    # 🔥 修复: .flatten() 解决维度不匹配报错
    input_dim = processor.process_single_row(dummy_input).flatten().shape[0]
    
    print(f"✅ 模型输入特征维度已校准: {input_dim}")
    
    # 3. 初始化模型并移至对应设备
    model = CatalystTransformer(input_dim=input_dim).to(device)
    
    # 4. 加载权重
    # 🔥 修复: weights_only=False
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval() # 开启评估模式
    
    return model, processor, checkpoint.get('times', np.arange(61))

# 🔥🔥 关键修复：必须在这里执行函数，给 model 赋值！🔥🔥
model, processor, target_times = load_resources()

# ==========================================
# 5. Streamlit 界面
# ==========================================
st.set_page_config(page_title="SACs 类芬顿动力学预测系统", layout="wide")
st.title("🧪 (Real-AI) 单原子催化剂驱动的类芬顿反应动力学预测系统 V1")

# 检查模型是否加载成功
if model is None:
    st.error(f"❌ 找不到模型文件！请检查路径设置。\n\nModel: {MODEL_PATH}\nData: {DATA_PATH}")
    st.stop()

# --- 侧边栏 ---
st.sidebar.header("1. 反应条件设置")

# 动态获取选项
cat_options = processor.categories['Catalyst_Type']
poll_options = processor.categories['Pollutant']
ox_options = processor.categories['Oxidant']
anion_options = processor.categories['Anion_Type']

catalyst_type = st.sidebar.selectbox("催化剂", cat_options, index=0)
pollutant = st.sidebar.selectbox("污染物", poll_options, index=0)
oxidant = st.sidebar.selectbox("氧化剂", ox_options, index=0)
anion_type = st.sidebar.selectbox("共存阴离子", anion_options, index=0)

st.sidebar.header("2. 数值参数")
ph_val = st.sidebar.slider("pH 值", 1.0, 14.0, 7.0)
cat_conc = st.sidebar.number_input("催化剂浓度 (g/L)", 0.0, 5.0, 0.1)
poll_conc = st.sidebar.number_input("污染物浓度 (mg/L)", 0.0, 100.0, 10.0)
pms_conc = st.sidebar.number_input("氧化剂浓度 (g/L)", 0.0, 10.0, 0.15)
anion_conc = st.sidebar.number_input("阴离子浓度 (mM)", 0.0, 100.0, 0.0)
temp_val = st.sidebar.number_input("温度 (K)", 273.0, 373.0, 298.0)

st.sidebar.markdown("---")
# 显示当前硬件状态
st.sidebar.caption(f"⚡️ Computing Device: **{str(device).upper()}**")

run_btn = st.sidebar.button("🚀 运行 Transformer 推理")

# --- 主显示区 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.info(f"✅ 模型状态: 在线 ({device})")
    input_dict = {
        'Catalyst': catalyst_type,
        'Pollutant': pollutant,
        'Oxidant': oxidant,
        'pH': ph_val,
        'T(K)': temp_val
    }
    st.write("当前输入摘要：")
    st.table(pd.DataFrame(input_dict, index=[0]).T)

with col2:
    if run_btn:
        try:
            # 1. 构建完整的输入字典
            full_input = {
                'Catalyst_Type': catalyst_type,
                'Pollutant': pollutant,
                'Oxidant': oxidant,
                'Anion_Type': anion_type,
                'pH': ph_val,
                'Catalyst_Conc': cat_conc,
                'Oxidant_Conc': pms_conc,
                'Pollutant_Conc_mgL': poll_conc,
                'Anion_Conc_mM': anion_conc,
                'Temp_K': temp_val
            }
            
            # 2. 预处理 (CPU -> Tensor -> Device)
            # 🔥 修复: .flatten() 确保维度正确
            feature_vec = processor.process_single_row(full_input).flatten()
            feature_tensor = torch.tensor(feature_vec, dtype=torch.float32).unsqueeze(0).to(device)
            
            # 3. 准备 Decoder 输入
            seq_len = len(target_times)
            tgt_input = torch.full((1, seq_len, 1), 0.5).to(device)
            
            # 4. 推理
            with torch.no_grad():
                output = model(feature_tensor, tgt_input)
                pred_curve = output.cpu().numpy().flatten()
            
            # 5. 后处理 (物理锁)
            pred_curve[0] = 1.0
            for i in range(1, len(pred_curve)):
                if pred_curve[i] > pred_curve[i-1]: 
                    pred_curve[i] = pred_curve[i-1]
            
            # 6. 绘图
            st.success("预测完成！")
            
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(target_times, pred_curve, 'r-o', linewidth=2, label='AI Prediction')
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("C/C0")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{catalyst_type} degrading {pollutant}")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
            # 7. 下载数据
            df_res = pd.DataFrame({"Time (min)": target_times, "Predicted C/C0": pred_curve})
            st.dataframe(df_res.T)
            
        except Exception as e:
            st.error(f"推理错误: {e}")
            st.write(e) # 打印详细错误信息以便调试
    else:
        st.info("👈 请在左侧调整参数，然后点击运行")