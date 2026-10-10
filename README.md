# 🌸 Flower Species Classification Web App

An end-to-end computer vision web application built with **PyTorch**, **ResNet-50**, and **Streamlit** to classify **102 distinct species of flowers** using the Oxford Flowers-102 dataset.

---

## 📸 Features

- **High Accuracy Transfer Learning:** Fine-tuned ResNet-50 backbone pre-trained on ImageNet.
- **Top-3 Predictions & Confidence Scores:** Visual probability breakdown with progress bars for predicted flower species.
- **Interactive UI:** Simple, intuitive web browser interface powered by Streamlit.
- **Corrected Label Indexing:** Seamless mapping of network logit indices directly to class names without off-by-one offsets.

---

## 🛠️ Project Structure

```text
├── app_cv.py             # Streamlit web application interface & inference logic
├── train_cv_model.py     # Training script with 2-stage fine-tuning pipeline
├── flower_resnet50.pth   # Saved model weights (generated after training)
└── README.md             # Project documentation
