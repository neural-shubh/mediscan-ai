# MediScan AI 🧠

Neural medical image classifier with Grad-CAM explainability.

## Live Demo
Deployed on Render.com

## Models
- **Brain MRI**: EfficientNetV2B3 — 4 class (Glioma / Meningioma / No Tumor / Pituitary) — 95% val accuracy
- **Chest X-Ray**: DenseNet121 — Binary (Normal / Pneumonia)
- [Separately deployed on ](https://huggingface.co/neural-shubh/mediscan-models/tree/main)

## Tech Stack
- **Backend**: FastAPI + TensorFlow
- **Frontend**: HTML + CSS + JS (no frameworks)
- **Models**: Hosted on Hugging Face Hub
- **Deployment**: Render.com

## Datasets
- [Chest X-Ray (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [Brain Tumor MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Features
- Grad-CAM heatmap visualization
- Confidence thresholding with radiologist warning
- Drag & drop image upload
- Responsive design

## ⚠️ Disclaimer
For research purposes only. Not a substitute for professional medical diagnosis.
