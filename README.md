---
title: PancreScan
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.42.0"
app_file: src/ui/app.py
pinned: false
---

# PancreScan: AI-Powered Pancreatic Cancer Detection

PancreScan is a deep learning research project designed to detect pancreatic cancer from CT scan images. It utilizes an ensemble of state-of-the-art Convolutional Neural Networks (CNNs) and segmentation models to achieve high diagnostic accuracy and precise structural highlighting.

## 🚀 Models

The project leverages four powerful pretrained architectures, fine-tuned for medical imaging:

*   **EfficientNet-V2-S**: A modern, efficient model optimized for training speed and parameter efficiency.
*   **DenseNet121**: A classic, dense connectivity architecture known for feature reuse.
*   **ConvNeXt-Tiny**: A "modernized" ResNet that competes with Vision Transformers (ViTs) in performance.
*   **UNet (Multi-Task Segmentation)**: A specialized architecture for medical image segmentation, newly added to generate structural masks and evaluate Dice/IoU metrics.

## 📁 Project Structure

*   `src/ui/`: Streamlit frontend application.
*   `src/api/`: FastAPI backend for running model inference.
*   `src/training/`: Scripts for training models, including K-Fold Cross Validation and UNet training.
*   `src/models/`: Neural network architecture definitions.
*   `outputs/`: Generated reports, metrics, and checkpoints.
*   `docker-compose.yml`: Docker configuration for spinning up the full stack.

## 🛠️ Installation & Execution

### 1. Using Docker (Recommended)

The easiest way to run both the FastAPI backend and Streamlit frontend is via Docker Compose.

```bash
docker-compose up --build
```
*   **Frontend (Streamlit)**: http://localhost:8501
*   **Backend (FastAPI)**: http://localhost:8000/docs

### 2. Manual Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RangeshPandianPT/PancreScan.git
    cd PancreScan
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+ installed.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the API:**
    ```bash
    uvicorn src.api.pancre_scan_api:app --host 0.0.0.0 --port 8000
    ```

4.  **Run the UI:**
    ```bash
    streamlit run src/ui/app.py
    ```

## 📊 Training Usage

### 1. K-Fold Cross-Validation
Run 5-fold cross-validation for classification models.

```bash
python src/training/run_kfold_cv.py --model efficientnet_v2_s --k-folds 5 --epochs 20 --output-dir outputs/my_experiment
```

### 2. UNet Segmentation Training
Train the UNet model with Dice and IoU metrics.

```bash
python src/training/train_unet.py
```

### 3. Ensemble Training
Train a weighted ensemble of two models to improve performance further.

```bash
python src/training/train_ensemble_smart.py --model-a densenet121 --model-b efficientnet_v2_s
```

## 📂 Output

Training results, including metrics (Accuracy, F1-Score, Recall, Dice, IoU), confusion matrices, and loss curves, are saved in the `outputs/` directory. Each run generates detailed JSON metrics, plots, and PDF summary reports.

## 📈 Results Overview

Preliminary results show strong performance across all architectures, with classification models achieving **>98% accuracy** on validation folds, and the new UNet model providing robust segmentation masks.
