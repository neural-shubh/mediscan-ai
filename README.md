# MediScan AI — Neural Medical Image Classifier

> AI-powered medical image analysis with Grad-CAM explainability — deployed live on Streamlit Cloud.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)](https://mediscan-ai-grcgwmxnbs4wonphcb8jwk.streamlit.app/)
[![Models](https://img.shields.io/badge/Models-Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface)](https://huggingface.co/neural-shubh/mediscan-models)
[![GitHub](https://img.shields.io/badge/Code-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/neural-shubh/mediscan-ai)

---

## 🌐 Live Demo

**[https://mediscan-ai-grcgwmxnbs4wonphcb8jwk.streamlit.app](https://mediscan-ai-grcgwmxnbs4wonphcb8jwk.streamlit.app)**

Upload a Brain MRI or Chest X-Ray image and get an instant AI-powered diagnosis with visual explainability.

---

## 🩺 What It Does

MediScan AI classifies medical images across two modalities:

### 🧠 Brain Tumor MRI
Detects and classifies brain tumors into 4 categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

**Model**: EfficientNetV2B3 (pretrained on ImageNet, fine-tuned on brain MRI)
**Validation Accuracy**: ~95%

### 🫁 Chest X-Ray (Pneumonia Detection)
Binary classification:
- **Normal**
- **Pneumonia**

**Model**: DenseNet121 (pretrained on ImageNet, fine-tuned on chest X-rays)

---

## ✨ Features

- **Grad-CAM Heatmaps** — visualizes exactly where the model looks to make its prediction
- **Confidence Scoring** — shows prediction confidence with color-coded indicators
- **Low Confidence Warning** — flags uncertain predictions and recommends radiologist consultation
- **Dual Modality** — supports both Brain MRI and Chest X-Ray in one unified interface
- **Dark Medical UI** — clean, professional interface built for medical imaging context

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend/App | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Brain Model | EfficientNetV2B3 |
| Chest Model | DenseNet121 |
| Explainability | Grad-CAM |
| Model Hosting | Hugging Face Hub |
| Deployment | Streamlit Community Cloud |

---

## 🏗 Architecture

```
User uploads image
       ↓
Streamlit Frontend
       ↓
Preprocessing (EfficientNet / DenseNet preprocess)
       ↓
Model Inference (loaded from Hugging Face Hub)
       ↓
Grad-CAM Heatmap Generation
       ↓
Results: Prediction + Confidence + Heatmap
```

---

## 📊 Model Training

### Brain Tumor MRI
- **Dataset**: [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — 7,023 images
- **Base Model**: EfficientNetV2B3 (frozen, ImageNet weights)
- **Head**: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(4, Softmax)
- **Training**: 2-phase — frozen base then fine-tuned
- **Augmentation**: Rotation, zoom, horizontal/vertical flip
- **Class Weights**: Applied to handle imbalance
- **Val Accuracy**: ~95%

### Chest X-Ray
- **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,216 images
- **Base Model**: DenseNet121 (frozen, ImageNet weights)
- **Head**: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.4) → Dense(1, Sigmoid)
- **Loss**: Binary Crossentropy
- **Augmentation**: Shear, zoom, horizontal flip

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/neural-shubh/mediscan-ai.git
cd mediscan-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
mediscan-ai/
├── app.py              ← Streamlit application
├── requirements.txt    ← Python dependencies
├── runtime.txt         ← Python version (3.11)
└── README.md
```

Models are hosted on Hugging Face and downloaded automatically at runtime:
- `neural-shubh/mediscan-models/brain_model_final.keras`
- `neural-shubh/mediscan-models/chest_model_final.keras`

---

## ⚠️ Disclaimer

This project is for **research and educational purposes only**.
It is **not** intended for clinical use and should **not** be used as a substitute for professional medical diagnosis.
Always consult a qualified radiologist or medical professional for medical decisions.

---

## 👨‍💻 Author

**Shubh** — [@neural-shubh](https://github.com/neural-shubh)

---

*Built with TensorFlow, Streamlit, and a lot of debugging 🔥*
