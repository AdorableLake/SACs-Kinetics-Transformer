## build_dataset_v3.py
import pandas as pd
import numpy as np
import torch
import os
import shutil
from datetime import datetime
from data_encoder import CatalystFeatureProcessor

# --- 1. 配置 ---
METADATA_PATH = 'metadata.xlsx'
RAW_DATA_DIR = './data_raw'
PROCESSED_DIR = './processed_data'
# 输出文件名的前缀
OUTPUT_PREFIX = 'catalyst_dataset_v3'

# 🔥 核心：通用时间网格 (13个点)
# 涵盖了 Gao (0-3), Cheng (0-8), Xu (0-60) 的所有特征区间
TARGET_TIMES = np.array([0, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60], dtype=np.float32)

def find_file(filename, search_path):
    """递归查找文件"""
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def build_dataset():
    print(f"🚀 启动 V3 数据融合引擎...")
    
    # 1. 读取 Excel
    try:
        df = pd.read_excel(METADATA_PATH)
        print(f"✅ 读取元数据: {len(df)} 条记录")
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 2. 拟合编码器
    processor = CatalystFeatureProcessor()
    
    # 建立映射
    col_map = {
        'Catalyst_Type': 'Catalyst Type', 'Pollutant': 'Pollutant', 'Oxidant': 'PMS',
        'Anion_Type': 'Anion Type', 'pH': 'pH', 'Catalyst_Conc': 'Catalyst Conc',
        'Oxidant_Conc': 'PMS Conc', 'Pollutant_Conc_mgL': 'Pollutant Conc',
        'Anion_Conc_mM': 'Anion Conc', 'Temp_K': 'Temp'
    }
    
    # 准备 Fit 数据
    fit_data = []
    for _, row in df.iterrows():
        item = {}
        for code_key, excel_col in col_map.items():
            val = row.get(excel_col)
            if pd.isna(val): val = 0
            item[code_key] = val
        fit_data.append(item)
    
    processor.fit(pd.DataFrame(fit_data))
    print("✅ 特征处理器拟合完成。")

    # 3. 数据融合与插值
    X_list = []
    y_list = []
    valid_files = []
    missing_count = 0

    print("⚡️ 开始执行多源数据融合 (Interpolation)...")
    
    for idx, row in df.iterrows():
        filename = row['File Name']
        if not filename.endswith('.csv'): filename += '.csv'
        
        file_path = find_file(filename, RAW_DATA_DIR)
        
        if not file_path:
            print(f"⚠️  [缺失] 找不到文件: {filename}")
            missing_count += 1
            continue
            
        try:
            # 读取原始曲线
            csv_data = pd.read_csv(file_path, header=None)
            original_times = csv_data[0].values
            original_concs = csv_data[1].values
            
            # 线性插值 (核心魔法)
            # Gao的数据(3min结束)会被自动延展，Xu的数据(60min)会被保留长尾
            interpolated_concs = np.interp(TARGET_TIMES, original_times, original_concs)
            y_seq = interpolated_concs.astype(np.float32)
            
        except Exception as e:
            print(f"❌ [错误] 读取失败 {filename}: {e}")
            continue

        # 编码特征
        feature_dict = fit_data[idx]
        X_vec = processor.process_single_row(feature_dict).flatten()

        X_list.append(X_vec)
        y_list.append(y_seq)
        valid_files.append(filename)

    # 4. 打包保存
    if len(X_list) == 0: 
        print("❌ 没有有效数据，退出。")
        return

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

    print(f"\n📊 V3 数据集报告:")
    
    # 自动从文件名中提取作者前缀 (假设文件名格式为 "author_xxx.csv")
    prefixes = set([f.split('_')[0] for f in valid_files])
    
    print(f"\n📊 V3 数据集报告:")
    print(f"   来源文献数: {len(prefixes)} ({', '.join(prefixes)})") # <--- 变聪明了

    print(f"   有效样本数: {len(X_list)}")
    print(f"   缺失文件数: {missing_count}")
    print(f"   时间网格: 0 -> 60 min (13 points)")
    print(f"   Tensor形状: X={X_tensor.shape}, y={y_tensor.shape}")

    dataset = {
        'X': X_tensor,
        'y': y_tensor,
        'filenames': valid_files, # 保存文件名以便追踪
        'processor': processor,
        'times': TARGET_TIMES
    }
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # --- 双重保存 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 历史存档
    archive_path = os.path.join(PROCESSED_DIR, f'{OUTPUT_PREFIX}_{timestamp}.pt')
    torch.save(dataset, archive_path)
    
    # 2. 最新版 (供 train_v3.py 读取)
    latest_path = os.path.join(PROCESSED_DIR, f'{OUTPUT_PREFIX}_latest.pt')
    torch.save(dataset, latest_path)
    
    print(f"🎉 数据集保存完毕:")
    print(f"   👉 存档: {archive_path}")
    print(f"   👉 最新: {latest_path}")

if __name__ == "__main__":
    build_dataset()