# Convolutional Neural Network Framework for Handwritten Digit Detection
A robust deep learning framework for accurate **handwritten digit recognition** using **Convolutional Neural Networks (CNNs)**.

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-CNN-red)](#architecture)
[![Accuracy](https://img.shields.io/badge/Accuracy-99%2B%25-brightgreen)](#performance)

</div>

---

📋 **Overview**  
This repository implements a comprehensive **handwritten digit recognition system** using **Convolutional Neural Networks (CNNs)**. The framework achieves **over 99% accuracy** on the MNIST dataset and provides tools for **training, evaluation, and real-time prediction**. Designed with **modularity and extensibility** in mind, it supports various CNN architectures and preprocessing techniques.

---

✨ **Features**

- **Multiple Input Modalities:** Support for canvas drawings, image uploads, and real-time camera input.  
- **High Accuracy:** Achieves **99%+ accuracy** on the MNIST benchmark.  
- **Real-time Inference:** Fast prediction with **confidence scoring**.  
- **Data Augmentation:** Built-in preprocessing and augmentation pipelines.  
- **Model Comparison:** Tools for evaluating **different CNN architectures**.  
- **Visualization:** Includes **training curves, confusion matrices, and feature maps**.  
- **Export Capabilities:** Save results in **CSV, Excel, or PDF formats**.

## 🚀 Quick Start

### Prerequisites
- **Python 3.8** or higher  
- **TensorFlow 2.x**  
- **Git**

---

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/handwritten-digit-detection.git
cd handwritten-digit-detection
```
## 📌 Basic Usage

### Train a New Model

```bash
python train_model.py \
    --epochs 20 \
    --batch_size 32 \
    --model_name cnn_digit_model
```
### Evaluate the Model
```bash
python evaluate_model.py \
    --model_path models/cnn_digit_model.h5
```
### Make Predictions
```bash
from predict import DigitPredictor
import numpy as np

# Initialize predictor
predictor = DigitPredictor("models/cnn_digit_model.h5")

# Predict from an image file
digit, confidence = predictor.predict_from_image("data/custom/test_5.png")
print(f"Predicted: {digit} with {confidence:.2%} confidence")

# Predict from a canvas drawing (NumPy array)
canvas_image = np.random.rand(28, 28)  # Replace with actual canvas data
digit, confidence = predictor.predict_from_canvas(canvas_image)
print(f"Predicted: {digit} with {confidence:.2%} confidence")
```

## 📊 Model Architecture

The core **Convolutional Neural Network (CNN)** architecture is designed for efficient and accurate handwritten digit recognition. The model processes grayscale images of size **28×28** and outputs probabilities for **10 digit classes (0–9)**.

### Architecture Flow

```text
Input (28×28×1)
→ Conv2D (32 filters, 3×3) → ReLU → MaxPooling (2×2)
→ Conv2D (64 filters, 3×3) → ReLU → MaxPooling (2×2)
→ Conv2D (64 filters, 3×3) → ReLU
→ Flatten
→ Dense (64) → ReLU → Dropout (0.5)
→ Dense (10) → Softmax
```

### Key Components

Convolutional Layers: Extract spatial features from digit images
Max Pooling: Reduce spatial dimensions, increase translation invariance
Dropout: Prevent overfitting (50% dropout rate)
Softmax Output: 10-class classification (digits 0-9)

### 🧪 Training Details
### Dataset
The model is trained on the MNIST dataset:

60,000 training images

10,000 testing images

28×28 grayscale images

10 classes (digits 0-9)

### Data Augmentation
To improve generalization, the following augmentations are applied:

-Random rotations (±10 degrees)
-Random zoom (90-110%)
-Random shifts (±10%)
-Brightness adjustments

### Training Configuration
```bash
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "categorical_crossentropy",
    "validation_split": 0.2,
    "early_stopping_patience": 5
```
## 📈 Performance

The following table summarizes the performance of different models evaluated on the **MNIST dataset**. Metrics include accuracy, precision, recall, F1-score, and average inference time per sample.

| Model            | Accuracy | Precision | Recall | F1-Score | Inference Time |
|------------------|----------|-----------|--------|----------|----------------|
| CNN              | 99.2%    | 99.1%     | 99.2%  | 99.1%    | 2–5 ms         |
| Random Forest    | 96.8%    | 96.5%     | 96.8%  | 96.6%    | 10–15 ms       |
| SVM              | 97.2%    | 97.0%     | 97.2%  | 97.1%    | 5–10 ms        |


