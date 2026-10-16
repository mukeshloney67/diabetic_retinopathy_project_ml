# 👁️ Diabetic Retinopathy Detection & Severity Classification

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://tensorflow.org/)
[![Architecture](https://img.shields.io/badge/Model-InceptionV3-red.svg)](https://arxiv.org/abs/1512.00567)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Clinical Scale](https://img.shields.io/badge/Grading-ICDR%20Standard-purple.svg)](https://www.aao.org/)

An automated clinical deep learning system for screening, detecting, and grading **Diabetic Retinopathy (DR)** from digital fundus photography using **InceptionV3 transfer learning and fine-tuning** in **TensorFlow/Keras**.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Clinical Severity Grading](#-clinical-severity-grading)
- [Architecture & Training Strategy](#-architecture--training-strategy)
- [Image Preprocessing & Enhancement](#-image-preprocessing--enhancement)
- [Project Directory Structure](#-project-directory-structure)
- [Installation](#-installation)
- [Quick Start & Usage](#-quick-start--usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Metrics](#3-evaluation--metrics)
  - [4. Single & Batch Inference](#4-single--batch-inference)
  - [5. Interactive Streamlit Web UI](#5-interactive-streamlit-web-ui)
- [Explainability with Grad-CAM](#-explainability-with-grad-cam)
- [Evaluation Metrics](#-evaluation-metrics)
- [License](#-license)

---

## 🔬 Overview

Diabetic Retinopathy is one of the leading causes of preventable blindness among working-age adults globally. Early detection through regular fundus photography screening can reduce severe vision loss by over 90%.

This project provides an end-to-end Machine Learning pipeline that takes raw fundus photography, applies **Ben Graham illumination normalization**, extracts deep hierarchical features using a fine-tuned **InceptionV3 engine**, and classifies the condition into the **5 international clinical stages**.

---

## 🏥 Clinical Severity Grading

Based on the **International Clinical Diabetic Retinopathy (ICDR)** severity scale:

| Stage | Severity | Key Clinical Findings | Action Required |
|:---:|:---|:---|:---|
| **0** | **No DR** | No abnormalities or lesions | Annual routine screening |
| **1** | **Mild NPDR** | Microaneurysms only | Follow-up in 6–12 months |
| **2** | **Moderate NPDR** | Microaneurysms, hard exudates, cotton wool spots | Follow-up in 3–6 months |
| **3** | **Severe NPDR** | >20 intraretinal hemorrhages in 4 quadrants, venous beading | Urgent ophthalmology referral (2–4 weeks) |
| **4** | **Proliferative DR (PDR)** | Neovascularization, vitreous/preretinal hemorrhage | Immediate specialist intervention (Laser/anti-VEGF) |

---

## 🧠 Architecture & Training Strategy

```
Input Fundus (299×299×3)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  InceptionV3 Backbone (Pretrained on ImageNet)        │
│  - Multi-scale convolutional factorizations            │
│  - Residual reduction blocks                           │
└────────────────────────────────────────────────────────┘
       │
       ▼
 Global Average Pooling 2D
       │
       ▼
 Dense(512, ReLU) ──► BatchNorm ──► Dropout(0.4)
       │
       ▼
 Dense(256, ReLU) ──► BatchNorm ──► Dropout(0.3)
       │
       ▼
 Dense(5, Softmax) ──► Class Probabilities [0 to 4]
```

### Two-Phase Training Strategy

1. **Phase 1: Transfer Learning (Feature Extraction)**
   - Backbone InceptionV3 layers are **frozen**.
   - Custom classification head is trained using Adam optimizer ($lr = 10^{-3}$).
   - Learns high-level retinopathy disease representations without disrupting ImageNet feature weights.

2. **Phase 2: Fine-Tuning**
   - Upper Inception blocks (from layer 249 onwards) are **unfrozen**.
   - Entire network is trained end-to-end with a lower learning rate ($lr = 10^{-5}$).
   - Adapts specific receptive fields directly to ophthalmological lesion patterns.

3. **Handling Medical Imbalance**
   - Automatically computes **inverse-frequency class weights** to ensure rare severe stages (Stage 3 & 4) receive proportional gradient updates.

---

## 🎨 Image Preprocessing & Enhancement

Retinal imaging often suffers from non-uniform illumination, variable camera exposure, and black peripheral border noise:

- **Circular ROI Cropping**: Automatically locates and crops out non-informative black outer borders.
- **Ben Graham Illumination Normalization**: Subtracts the local Gaussian-blurred average color to standardize exposure:
  $$\text{Enhanced} = 4 \times \text{Image} - 4 \times \text{GaussianBlur}(\text{Image}, \sigma=10) + 128$$
- **InceptionV3 Normalization**: Rescales pixel values to $[-1, 1]$.

---

## 📁 Project Directory Structure

```text
diabetic_retinopathy_project_ml/
├── config/
│   └── config.yaml              # Hyperparameters, augmentation & path settings
├── src/
│   ├── __init__.py
│   ├── config_loader.py         # Config parser & validator
│   ├── preprocessing.py         # Ben Graham filter & circular ROI cropping
│   ├── data_loader.py           # Augmentation generators & class weight balancer
│   ├── model.py                 # InceptionV3 architecture & 2-phase builder
│   ├── train.py                 # Two-phase training pipeline & checkpointing
│   ├── evaluate.py              # QWK, ROC curves, confusion matrix & metrics
│   ├── predict.py               # Single & batch inference engine
│   ├── gradcam.py               # Grad-CAM explainable AI heatmap generator
│   └── utils.py                 # Reproducibility seed & visualization tools
├── app/
│   └── app.py                   # Interactive Streamlit Web UI
├── scripts/
│   ├── prepare_data.py          # Stratified train/val/test data organizer
│   ├── generate_synthetic_data.py # Synthetic sample generator for testing
│   └── run_training.sh          # One-click training launcher script
├── notebooks/
│   └── 01_diabetic_retinopathy_exploration.ipynb
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # Unit tests for preprocessing & models
├── models/                      # Checkpoints directory (.gitkeep)
├── data/                        # Dataset directory (.gitkeep)
├── logs/                        # TensorBoard & evaluation artifacts (.gitkeep)
├── requirements.txt             # Python dependencies
├── setup.py                     # Python package installer
├── .gitignore                   # Excludes weights and large datasets
└── README.md                    # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/mukeshloney67/diabetic_retinopathy_project_ml.git
cd diabetic_retinopathy_project_ml
```

### 2. Set up virtual environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quick Start & Usage

### 1. Data Preparation

#### Using a real dataset (e.g. APTOS 2019 / EyePACS):
```bash
python scripts/prepare_data.py \
    --csv_path /path/to/train.csv \
    --images_dir /path/to/train_images \
    --output_dir data/ \
    --val_split 0.15 \
    --test_split 0.15
```

#### Or generate a synthetic dataset for testing:
```bash
python scripts/generate_synthetic_data.py --samples_per_class 20
```

---

### 2. Model Training

Run the two-phase training pipeline:

```bash
# Full two-phase training (Transfer Learning + Fine-Tuning)
python -m src.train --config config/config.yaml --phase both

# Or run Phase 1 only
python -m src.train --config config/config.yaml --phase phase1

# Or run via shell script
bash scripts/run_training.sh
```

---

### 3. Evaluation & Metrics

Evaluate trained checkpoints against the test set:

```bash
python -m src.evaluate --config config/config.yaml --model_path models/best_model.keras
```

Outputs:
- **Accuracy, Weighted F1, Macro F1**
- **Quadratic Weighted Kappa (QWK)** (competition standard)
- **Multi-Class ROC Curves** (`logs/roc_curve.png`)
- **Confusion Matrix** (`logs/confusion_matrix.png`)
- **Full Report** (`logs/evaluation_report.json`)

---

### 4. Single & Batch Inference

#### Single image prediction:
```bash
python -m src.predict --image path/to/retina.png --model models/best_model.keras
```

```text
=======================================================
👁️  DIABETIC RETINOPATHY PREDICTION RESULT
=======================================================
File:            retina_sample.png
Predicted Stage: Moderate NPDR (Class 2)
Severity:        Moderate Non-Proliferative
Confidence:      89.42%
Clinical Action: Follow up examination in 3-6 months; optimize glycemic control.
Key Findings:    Microaneurysms, hemorrhages, hard exudates, or cotton wool spots.

Probability Distribution:
  No DR               :  1.20% 
  Mild NPDR           :  5.14% █
  Moderate NPDR       : 89.42% ██████████████████████
  Severe NPDR         :  3.12% 
  Proliferative DR    :  1.12% 
=======================================================
```

#### Batch prediction on a folder:
```bash
python -m src.predict --dir path/to/test_images/ --output_csv predictions.csv
```

---

### 5. React.js Modern Web Interface (Vite + React 18)

A responsive, high-performance web interface built with **React 18** and **Vite** featuring drag-and-drop image uploading, live photometric statistics, accuracy verification against ground-truth stages, and batch cohort analysis.

```bash
# 1. Navigate to frontend and install dependencies
cd frontend
npm install

# 2. Start the development server
npm run dev
```
Open **`http://localhost:3000`** in your browser.

*(Optional) Start the FastAPI backend server to serve live InceptionV3 predictions directly:*
```bash
uvicorn api.main:app --reload --port 8000
```

#### Key Capabilities:
- **📸 Single Image Diagnostic & Retinal Statistics**:
  - Drag-and-drop fundus photograph upload (PNG, JPG, TIFF).
  - Photometric statistics: **Sharpness Index (Laplacian Variance)**, **Retinal FOV Coverage %**, **Mean Luminance**, **RMS Contrast**, and **Uniformity Ratio**.
  - **Clinical Grading & Accuracy**: Real-time evaluation against user-selected ground truth (Exact Match vs Stage Discrepancy & Quadratic Score).
  - **RGB Spectral Histograms**: Multi-channel curve visualization for vascular & hemorrhage profiling.
- **📂 Batch Upload & Cohort Accuracy Testing**:
  - Upload multiple retinal fundus images simultaneously.
  - Automatically extracts ground truth from filenames.
  - Computes cohort accuracy (%), average sharpness, gradability rates, and exports a structured CSV report.
- **🖼️ Preloaded Clinical Test Gallery**:
  - Test all 5 ICDR clinical stages with 1 click without needing local image files.

---

### 6. Streamlit Alternative Dashboard

```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.


### 6. Command-Line Statistics & Accuracy Inspector

You can also run image statistics and accuracy checks directly from your terminal:

```bash
python scripts/analyze_image.py --image path/to/retina.png --ground_truth 2
```

---

## 🔍 Explainability with Grad-CAM

For medical diagnostic AI, model interpretability is paramount. This project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** targeting InceptionV3's `mixed10` convolutional feature layer to highlight the precise retinal coordinates responsible for classification.

---

## 📊 Evaluation Metrics

Target performance on standard benchmarks (e.g. APTOS 2019 Blindness Detection):

- **Quadratic Weighted Kappa (QWK)**: `~0.82 - 0.88`
- **Multi-class AUC-ROC (Macro)**: `~0.92 - 0.95`
- **Overall Accuracy**: `~82% - 86%`

---

## 🧪 Running Unit Tests

Run test suites to verify preprocessing, data loaders, and models:

```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Mukesh Kumar**  
GitHub: [@mukeshloney67](https://github.com/mukeshloney67)
