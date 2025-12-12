# data_encoder.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

class CatalystFeatureProcessor:
    def __init__(self):
        # 1. 定义分类特征的"词表"
        self.categories = {
            'Catalyst_Type': ['Fe-N2/C', 'Fe-N3/C', 'Fe-N4/C', 'NC', 'Other', 
                              'CN', 'FeNano-CN', 'FeSA-CN',
                              'SA-Fe-NC', 'Fe-NC', 'CS',
                              'Zn-N@C-5', 'Zn-N@C-10', 'Zn-N@C-20', 'C',
                              'ZIF-CoN3P-C', 'ZIF-CoN4-C', 'ZIF-P-C', 'ZIF-C',
                              'SA-Mn-NC', 'Mn-NC', 'nMn', 'Mn2+', 'SA-Mn-NC-600', 'SA-Mn-NC-1000',
                              'Co-N2', 'Co-N3', 'Co-N4', 'Co-NPs',
                              'SNC_CoSA-0.4', 'SNC_CoSA-0.1', 'SNC_CoSA-0.05', 'SNC_CoSA-0.025', 'SNC_CoSA-0.01','SNC_NiSA-0.05', 'SNC_CuSA-0.05', 'SNC_FeSA-0.05','Metal-free SNC', 'NC_CoSA-0.05',
                              'Fe-SAC', 'Co-SAC', 'Cu-SAC'],

            'Pollutant': ['TC', 'BPA', 'Rhodamine B', 'AO7', 'PM', 'CIP', 
                          'SSZ',
                          '2-CP', 'DP',
                          'SMX',
                          'SDZ', 'SMZ', 'OFX', 'ERY', 'TET', 'OTET', 'CLD',
                          '4-CP', 'NB', 'RhB',
                          'MB', 'BA', 'CMZP',
                          'CBZ', 'CP', 'CPL', 'CTC', 'HBA', 'NBA', 'NP', 'NPX', 'PCM', 'SMT', 'SN'],

            'Oxidant': ['PMS', 'PDS', 'H2O2'],

            'Anion_Type': ['None', 'Cl-', 'HCO3-', 'NO3-', 'SO4--', 'HA',
                           'H2PO4-',
                           'HPO4--']
        }
        
        # 2. 初始化编码器
        self.encoders = {}
        for col, cats in self.categories.items():
            self.encoders[col] = OneHotEncoder(categories=[cats], handle_unknown='ignore', sparse_output=False)
            
        # 3. 数值特征归一化器
        self.scaler = StandardScaler()
        
        # 定义数值列名 (内部使用下划线)
        self.num_cols = ['pH', 'Catalyst_Conc', 'Oxidant_Conc', 'Pollutant_Conc_mgL', 'Anion_Conc_mM', 'Temp_K']

    def fit(self, df):
        """
        根据训练数据统计数值特征的均值和方差，并激活分类编码器
        """
        self.scaler.fit(df[self.num_cols])
        
        for col, cats in self.categories.items():
            # 修复 Numpy 字符串兼容性问题
            dummy_data = np.array(cats, dtype=object).reshape(-1, 1)
            self.encoders[col].fit(dummy_data)

        print("Feature Processor Fitting Complete (Scaler + Encoders).")

    def process_single_row(self, row_dict):
        """
        处理单行数据 - 鲁棒性增强版
        """
        # 1. 处理数值特征 (关键修改：自动补全缺失键)
        safe_num_dict = {}
        for col in self.num_cols:
            # 如果字典里没有这个键，就填 0.0
            safe_num_dict[col] = row_dict.get(col, 0.0)
            
        num_df = pd.DataFrame([safe_num_dict])[self.num_cols]
        num_vec = self.scaler.transform(num_df) 
        
        # 2. 处理分类特征
        cat_vecs = []
        for col in self.categories.keys():
            val = row_dict.get(col, 'None')
            if pd.isna(val) or val == 0:
                val = 'None'
            else:
                val = str(val)
                
            vec = self.encoders[col].transform(np.array([[val]], dtype=object))
            cat_vecs.append(vec)
            
        # 3. 拼接
        final_vec = np.hstack([num_vec] + cat_vecs)
        return final_vec

    def get_feature_dim(self):
        return len(self.num_cols) + sum([len(c) for c in self.categories.values()])

# --- 自测代码 ---
if __name__ == "__main__":
    p = CatalystFeatureProcessor()
    
    # 模拟 build_dataset 里的 fit 数据
    df = pd.DataFrame([{
        'pH': 7, 'Catalyst_Conc': 0.2, 'Oxidant_Conc': 0.2, 
        'Pollutant_Conc_mgL': 20, 'Anion_Conc_mM': 0, 'Temp_K': 298
    }])
    p.fit(df)
    
    # 模拟一个“残缺”的输入（没有温度，没有浓度）
    # 以前这里会报错，现在应该能自动补0并通过
    test_row = {'Catalyst_Type': 'Fe-N3/C', 'pH': 7}
    
    vec = p.process_single_row(test_row)
    print("✅ 自测通过！")
    print("   即使输入缺少字段，代码也能自动补零处理。")
    print("   生成向量形状:", vec.shape)