<div align="center">

# 🧪 基于 Transformer 的单原子催化剂动力学智能预测系统


[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<!-- 语言切换按钮 -->
[![English](https://img.shields.io/badge/Language-English-blue)](README_EN.md)
[![Chinese](https://img.shields.io/badge/Language-中文-gray)](README.md)

---

> **这是一个融合 Transformer 深度学习架构与物理约束机制的 AI 框架，旨在实现单原子催化剂（SACs）降解有机污染物全过程动力学曲线的端到端预测。**

</div>

---

## 🌟 核心功能

*   **📈 端到端序列预测**: 突破传统机器学习仅能预测单一 $k$ 值的局限，利用 **Encoder-Decoder Transformer** 生成 0-60 分钟完整的动力学曲线。
*   **🧬 多模态特征嵌入**: 对催化剂微观结构（金属中心、配位环境）、环境因子（pH、共存阴离子）及污染物分子性质进行高维特征编码。
*   **⚛️ 物理信息约束**: 引入质量守恒与单调性修正机制 (Physics-Informed Constraints)，确保预测结果符合化学基本原理（无负浓度）。
*   **🚀 实时可视化交互**: 基于 **Streamlit** 构建的交互式 Web 系统，支持 **CUDA/MPS** 硬件加速推理。

---

## 🛠️ 系统架构

### 1. 数据处理流水线
针对多源异构文献数据的自动化 ETL（抽取、转换、加载）与时空对齐流程。

<div align="center">
  <img src="assets/data_pipeline.png" width="80%" alt="Data Pipeline">
</div>

### 2. 模型网络架构
基于自注意力机制 (`Self-Attention`) 定制的 `Transformer` 编码器-解码器（`Encoder-Decoder`）结构。

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

## 📊 性能表现
- R² Score: > 0.99 (测试集最佳表现)
- RMSE: < 0.03
- 硬件支持: 自动检测并调用 NVIDIA CUDA 或 Apple MPS (Metal Performance Shaders) 进行加速。

---

## 📷 系统截图
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

## 👨‍💻 作者
- Lake (AdorableLake)
- 🎓 天津大学 & 佐治亚理工学院环境工程双硕士
- 🎓 浙江理工大学工业设计学士
- 🔬 研究兴趣：科学智能（AI4S）、环境信息学、人机交互（HCI）

---

*免责声明：本项目是研究项目的一部分。数据可用性取决于发表状态。*