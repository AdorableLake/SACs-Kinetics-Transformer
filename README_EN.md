<div align="center">

# 🧪 AI-Driven Kinetics Prediction System for Single-Atom Catalysts (SACs)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

<!-- 语言切换按钮 -->
[![English](https://img.shields.io/badge/Language-English-blue)](README_EN.md)
[![Chinese](https://img.shields.io/badge/Language-中文-gray)](README.md)

> **A Deep Learning framework integrating Transformer architecture with physics-informed constraints to predict the full-process degradation kinetics of organic pollutants by Single-Atom Catalysts (SACs).**

</div>

---

## 🌟 Key Features

*   **📈 End-to-End Sequence Prediction**: Unlike traditional ML models that predict a single $k$ value, this model generates full 0-60 min kinetic curves using an **Encoder-Decoder Transformer**.
*   **🧬 Multi-Modal Feature Embedding**: Encodes catalyst structures (metal center, coordination), environmental factors (pH, anions), and pollutant properties.
*   **⚛️ Physics-Informed Constraints**: Incorporates mass conservation and monotonicity checks to ensure physically valid predictions (no negative concentrations).
*   **🚀 Real-Time Visualization**: Interactive Web App built with **Streamlit**, supporting hardware acceleration (CUDA/MPS).

## 🛠️ System Architecture

### 1. Data Pipeline
Automated ETL process for multi-source heterogeneous data alignment.

<div align="center">
  <img src="assets/data_pipeline_define.png" width="80%" alt="Data Pipeline">
</div>

### 2. Model Architecture
A customized Transformer with Self-Attention mechanisms.

<div align="center">
  <img src="assets/model_architecture_define.png" width="80%" alt="Model Architecture">
</div>

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/AdorableLake/SACs-Kinetics-Transformer.git
cd SACs-Kinetics-Transformer
pip install -r requirements.txt
```

### 2. Run the Web GUI
```bash
streamlit run app_real.py
```

---

## 📊 Performance
- R² Score: > 0.99 (on Test Set)
- RMSE: < 0.03
- Hardware Support: Auto-detects NVIDIA CUDA or Apple MPS (Metal Performance Shaders) for acceleration.

---

## 📷 Screenshots
- Interactive Dashboard
<div align="center">
<img src="assets/interactive_dashboard.png" width="100%" alt="Dashboard">
</div>

- Prediction Results (Best vs Worst Case)
<div align="center">
<img src="assets/result_v3.6_best_20251208_164608.png" width="48%" alt="Best Case">
<img src="assets/result_v3.6_worst_20251208_164608.png" width="48%" alt="Worst Case">
</div>

---

## 👨‍💻 Author
Lake (AdorableLake)
- 🎓 M.S. Environmental Engineering |  Georgia Tech & Tianjin Univ.
- 🎓 B.S. Industrial Design | Zhejiang Sci-Tech Univ.
- 🔬 Research Focus: AI for Science, Environmental Informatics, Human Computer Interactiion.

---

*Disclaimer: This project is part of a research study. Data availability subject to publication status.*
