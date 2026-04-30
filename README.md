# 🧠 Brain Tumor Classification Using Deep Learning (EfficientNetB0)

> Automated multi-class brain tumor classification from MRI images using transfer learning, with Grad-CAM explainability.

**Authors:** Hasan Akhtars & Nawaf Barebood  
**Course Project — April 2026**

---

## 📌 Overview

This project develops a deep learning pipeline to classify brain MRI scans into four categories:

| Class | Description |
|---|---|
| **Glioma** | Malignant tumor originating from glial cells |
| **Meningioma** | Typically benign tumor arising from the meninges |
| **Pituitary** | Tumor forming in the pituitary gland |
| **No Tumor** | Healthy brain scan (negative class) |

The model achieves **97.4% test accuracy** using EfficientNetB0 with transfer learning, and includes Grad-CAM heatmaps to visually explain predictions.

---

## 🗂️ Repository Structure

```
├── BrainTumorClassification.ipynb   # Main project notebook (report + results)
├── README.md                        # This file
```

---

## 📊 Results Summary

| Class | Test Accuracy |
|---|---|
| No Tumor | 100.0% |
| Pituitary | 99.3% |
| Meningioma | 95.9% |
| Glioma | 94.4% |
| **Overall** | **97.4%** |

- **Best Validation Accuracy:** ~99.0%  
- **Training Time:** ~28 minutes on NVIDIA T4 GPU  
- **Model Parameters:** ~4.2M (EfficientNetB0)

---

## 🧪 Methods

- **Architecture:** EfficientNetB0 (pre-trained on ImageNet) with custom classification head
- **Training Strategy:** Two-phase transfer learning
  - Phase 1: Frozen backbone, train head only (epochs 1–5, lr=1e-4)
  - Phase 2: Unfreeze top 30 layers, end-to-end fine-tuning (epochs 6–15, lr=1e-5)
- **Augmentation:** Horizontal flip, rotation ±15°, zoom ±10%, color jitter
- **Explainability:** Grad-CAM heatmaps on the last convolutional block
- **Framework:** PyTorch / TensorFlow (Google Colab, T4 GPU)

---

## 📦 Dataset

**Brain Tumor MRI Dataset** — Masoud Nickparvar, Kaggle 2021  
🔗 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

~7,023 T1-weighted MRI images across 4 classes. Split: 80% train / 10% val / 10% test (stratified).

> Download the dataset from Kaggle and place it in a `data/` directory before running the notebook.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision numpy matplotlib scikit-learn opencv-python Pillow
```

### Run the Notebook

```bash
jupyter notebook BrainTumorClassification.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/) for free GPU access.

---

## 🔍 Grad-CAM Visualizations

Grad-CAM heatmaps confirm the model attends to clinically correct regions:

- **Glioma** → Upper cerebral cortex where the mass is located
- **Meningioma** → Lateral inferior skull base (extra-axial location)
- **No Tumor** → Diffuse cortical patterns (no focal hotspot)
- **Pituitary** → Sella turcica (inferior-central region) — learned without spatial supervision

---

## 👥 Team Contributions

**Hasan Akhtars**
- Dataset preprocessing and augmentation pipeline
- EfficientNetB0 architecture and Phase 1 training
- Hyperparameter tuning and training curve analysis

**Nawaf Barebood**
- Phase 2 fine-tuning
- Grad-CAM implementation and heatmap analysis
- Confusion matrix, metrics, and report writing

---

## 📚 References

- Tan & Le (2019) — [EfficientNet](https://arxiv.org/abs/1905.11946), ICML 2019
- Selvaraju et al. (2017) — [Grad-CAM](https://arxiv.org/abs/1610.02391), ICCV 2017
- Goodfellow et al. (2016) — [Deep Learning](http://www.deeplearningbook.org), MIT Press
- Nickparvar (2021) — [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), Kaggle
