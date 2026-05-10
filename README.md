# 🧠 AI-Powered Brain MRI Tumor Segmentation

**Metadata-Infused Deep Learning Pipeline for Lower-Grade Glioma (LGG) Semantic Segmentation**

> AI Campus — Team 1 | Malav Champaneria, Amisha Rastogi, Jinal Panchal  
> Mentor: Dr. Iman Dehzangi | May 2026

---

## 📋 Overview

This project implements an end-to-end deep learning pipeline for brain tumor segmentation from MRI scans. It progressively builds from a standard U-Net to an **Attention-gated, Metadata-Infused U-Net** that fuses clinical patient metadata (age, gender, histology, tumor location, genomic clusters) with image features for improved segmentation accuracy.

### Key Results

| Model | Dice Score | IoU Score |
|-------|-----------|----------|
| BasicUNet | 0.8941 | 0.8101 |
| MetadataUNet | 0.8949 | 0.8111 |
| AttentionMetadataUNet | 0.8718 | 0.7756 |
| **AttentionMetadataUNet + Scheduler** | **0.8960** | **0.8135** |

---

## 🗂️ Project Structure

```
bioMed/
├── dashboard/                  # Streamlit web application
│   ├── app.py                  # Main dashboard (5-page interactive UI)
│   └── inference.py            # Model architectures & inference helpers
├── dashboard_data/             # Exported training artifacts
│   ├── models/                 # Trained model weights (.pth)
│   ├── gallery/                # Sample prediction images
│   ├── training_histories.json # Epoch-by-epoch metrics
│   ├── final_model_comparison.csv
│   ├── per_sample_scores.csv
│   ├── feature_mapping.json    # Metadata encoding schema
│   ├── dataset_stats.json
│   ├── tumor_coverage.csv
│   └── scaler.joblib           # Age StandardScaler
├── src/
│   └── AICampus_project.ipynb  # Training notebook (run on Colab)
├── AI_Campus_Team1_Report.docx # Final project report
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **pip** package manager
- **GPU (optional)**: CUDA-compatible GPU for faster inference; CPU works too

### 1. Clone the Repository

```bash
git clone https://github.com/Malav786/Brain_MRI_Tumor_Segmentation
cd bioMed
```

### 2. Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Model Weights

Model weights are not included in the repository due to size constraints (~130MB).

**Option A - Use preset data**

Place the downloaded files in `dashboard_data/models/`:
```
dashboard_data/models/
├── best_BasicUNet.pth
├── best_MetadataUNet.pth
├── best_AttentionMetadataUNet.pth
└── best_AttentionMetadataUNet_scheduler.pth
```

**Option B - Train from scratch:**
Open `src/AICampus_project.ipynb` in Google Colab (GPU runtime required) and run all cells. The notebook will download the dataset, train all 4 models, and export artifacts to `dashboard_data/`.

### 5. Run the Dashboard

```bash
cd dashboard
streamlit run app.py
```

The dashboard will open at **http://localhost:8501** with 5 interactive pages:
- 🏠 **Overview** — KPIs, architecture summary, model scores
- 🔬 **Live Inference** — Upload MRI + metadata → real-time segmentation
- 📊 **Training Analytics** — Interactive loss/Dice/IoU curves
- 🏆 **Model Comparison** — Per-sample distributions, per-patient breakdown
- 🖼️ **Prediction Gallery** — Visual MRI/mask/overlay comparison

---

## 🔬 Training (Google Colab)

The training notebook is designed to run on **Google Colab with GPU**:

1. Upload `src/AICampus_project.ipynb` to Google Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4/A100)
3. Run all cells sequentially
4. Download the generated `dashboard_data/` folder
5. Place it in the project root

### Dataset

The notebook automatically downloads the [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) via `kagglehub`:
- **7,858** MRI-mask pairs from **110** patients
- Binary segmentation masks for lower-grade glioma tumors
- Clinical metadata: age, gender, histological type, tumor location, genomic clusters

---

## 🏗️ Architecture

The pipeline implements 4 progressive model variants:

1. **BasicUNet** — Standard 3-level encoder-decoder with skip connections
2. **MetadataUNet** — Residual blocks + MLP-encoded clinical metadata fused at the bottleneck
3. **AttentionMetadataUNet** — Adds Attention Gates on decoder skip connections
4. **AttentionMetadataUNet + LR Scheduler** — ReduceLROnPlateau for better convergence

**Loss Function:** DiceBCE (Binary Cross-Entropy + Dice Loss)  
**Optimizer:** Adam (lr=1e-3)  
**Training:** 15 epochs, batch size 32, mixed-precision (AMP)

---

## 📦 Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch |
| Dashboard | Streamlit |
| Visualizations | Plotly, Matplotlib |
| Data Processing | NumPy, Pandas, OpenCV |
| ML Utilities | scikit-learn, joblib |
| Training Environment | Google Colab (A100 GPU) |

---

## 👥 Team

| Name | Role |
|------|------|
| Malav Champaneria | Lead Developer & ML Engineer |
| Amisha Rastogi | Data Analysis & Research |
| Jinal Panchal | Evaluation & Documentation |
| Dr. Iman Dehzangi | Mentor |

---

## 📄 License

This project is for academic purposes as part of the AI Campus program.

---

## 🙏 Acknowledgments

- [LGG MRI Segmentation Dataset](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) by Buda et al.
- [U-Net](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
- [Attention U-Net](https://arxiv.org/abs/1804.03999) — Oktay et al., 2018
