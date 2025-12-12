# 🧪 AI-Driven Kinetics Prediction System for Single-Atom Catalysts (SACs)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **A Deep Learning framework integrating Transformer architecture with physics-informed constraints to predict the full-process degradation kinetics of organic pollutants by Single-Atom Catalysts (SACs).**

---

## 🌟 Key Features

*   **📈 End-to-End Sequence Prediction**: Unlike traditional ML models that predict a single $k$ value, this model generates full 0-60 min kinetic curves using an **Encoder-Decoder Transformer**.
*   **🧬 Multi-Modal Feature Embedding**: Encodes catalyst structures (metal center, coordination), environmental factors (pH, anions), and pollutant properties.
*   **⚛️ Physics-Informed Constraints**: Incorporates mass conservation and monotonicity checks to ensure physically valid predictions (no negative concentrations).
*   **🚀 Real-Time Visualization**: Interactive Web App built with **Streamlit**, supporting hardware acceleration (CUDA/MPS).

## 🛠️ System Architecture

### 1. Data Pipeline
Automated ETL process for multi-source heterogeneous data alignment.
![System Architecture](assets/data_pipeline.png)

### 2. Model Architecture
A customized Transformer with Self-Attention mechanisms.
![System Architecture](assets/model_architecture.png)

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
![System Architecture](assets/interactive_dashboard.png)

- Prediction Results (Best vs Worst Case)
![System Architecture](assets/result_v3.6_best_20251208_164608.png)
![System Architecture](assets/result_v3.6_worst_20251208_164608.png)

---

## 👨‍💻 Author
Lake (AdorableLake)
- 🎓 M.S. in Environmental Engineering | Tianjin University & Georgia Tech
- 🎓 B.S. in Industrial Design | Zhejiang Sci-Tech University
- 🔬 Research Focus: AI for Science, HCI.

---

*Disclaimer: This project is part of a research study. Data availability subject to publication status.*