<div align="center">

# 🧪 基于 Transformer 的单原子催化剂动力学智能预测系统
### AI-Driven Kinetics Prediction System for Single-Atom Catalysts (SACs)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[🇺🇸 **English Version**](README_EN.md) | [🇨🇳 **中文说明**](README.md)

---

> **这是一个融合 Transformer 深度学习架构与物理约束机制的 AI 框架，旨在实现单原子催化剂（SACs）降解有机污染物全过程动力学曲线的端到端预测。**

</div>

---

## 🌟 核心功能 (Key Features)

*   **📈 端到端序列预测**: 突破传统机器学习仅能预测单一 $k$ 值的局限，利用 **Encoder-Decoder Transformer** 生成 0-60 分钟完整的动力学曲线。
*   **🧬 多模态特征嵌入**: 对催化剂微观结构（金属中心、配位环境）、环境因子（pH、共存阴离子）及污染物分子性质进行高维特征编码。
*   **⚛️ 物理信息约束**: 引入质量守恒与单调性修正机制 (Physics-Informed Constraints)，确保预测结果符合化学基本原理（无负浓度）。
*   **🚀 实时可视化交互**: 基于 **Streamlit** 构建的交互式 Web 系统，支持 **CUDA/MPS** 硬件加速推理。

---

## 🛠️ 系统架构 (System Architecture)

### 1. 数据处理流水线 (Data Pipeline)
针对多源异构文献数据的自动化 ETL（抽取、转换、加载）与时空对齐流程。

<div align="center">
  <img src="assets/data_pipeline.png" width="80%" alt="Data Pipeline">
</div>

### 2. 模型网络架构 (Model Architecture)
基于自注意力机制 (Self-Attention) 定制的 Transformer 编码器-解码器结构。

<div align="center">
  <img src="assets/model_architecture.png" width="60%" alt="Model Architecture">
</div>

---

## 🚀 快速开始 (Quick Start)

### 1. 安装依赖
```bash
git clone https://github.com/AdorableLake/SACs-Kinetics-Transformer.git
cd SACs-Kinetics-Transformer
pip install -r requirements.txt
```

## 2. 启动可视化系统
```bash
streamlit run app.py
```

---

## 📊 性能表现 (Performance)
- R² Score: > 0.99 (测试集最佳表现)
- RMSE: < 0.03
- 硬件支持: 自动检测并调用 NVIDIA CUDA 或 Apple MPS (Metal Performance Shaders) 进行加速。

---

## 📷 系统截图 (Screenshots)
- 交互式控制面板
<div align="center">
<img src="assets/interactive_dashboard.png" width="100%" alt="Dashboard">
</div>
- 预测结果对比 (最佳 vs 最差样本)
<div align="center">
<img src="assets/result_v3.6_best_20251208_164608.png" width="48%" alt="Best Case">
<img src="assets/result_v3.6_worst_20251208_164608.png" width="48%" alt="Worst Case">
</div>

---

## 👨‍💻 作者 (Author)
- Lake (AdorableLake)
- 🎓 M.S. in Environmental Engineering | Tianjin University & Georgia Tech
- 🎓 B.S. in Industrial Design | Zhejiang Sci-Tech University
- 🔬 Research Focus: AI for Science, HCI.

---

*Disclaimer: This project is part of a research study. Data availability subject to publication status.*