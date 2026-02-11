# PancreScan: AI-Powered Pancreatic Cancer Detection

PancreScan is a deep learning research project designed to detect pancreatic cancer from CT scan images. It utilizes an ensemble of state-of-the-art Convolutional Neural Networks (CNNs) to achieve high diagnostic accuracy.

## 🚀 Models

The project leverages three powerful pretrained architectures, fine-tuned for medical imaging:

*   **DenseNet121**: A classic, dense connectivity architecture known for feature reuse.
*   **EfficientNet-V2-S**: A modern, efficient model optimized for training speed and parameter efficiency.
*   **ConvNeXt-Tiny**: A "modernized" ResNet that competes with Vision Transformers (ViTs) in performance.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RangeshPandianPT/PancreScan.git
    cd PancreScan
    ```

2.  **Install dependencies:**
    Ensure you have Python 3.8+ installed. Install the required libraries (PyTorch, torchvision, pandas, scikit-learn, tqdm, matplotlib):
    ```bash
    pip install torch torchvision pandas scikit-learn tqdm matplotlib seaborn
    ```

## 📊 Usage

### 1. K-Fold Cross-Validation
Run 5-fold cross-validation for any of the supported models. This script handles data splitting, training, and report generation.

**Command Syntax:**
```bash
python run_kfold_cv.py --model [MODEL_NAME] --k-folds 5 --epochs 20 --freeze-backbone
```

**Supported Models:**
*   `densenet121`
*   `efficientnet_v2_s`
*   `convnext_tiny`
*   `efficientnet_b0` (Legacy)
*   `resnet50` (Legacy)

**Example:**
```bash
# Run efficientnet_v2_s
python run_kfold_cv.py --model efficientnet_v2_s --k-folds 5 --epochs 20 --output-dir outputs/my_experiment
```

### 2. Ensemble Training
Train a weighted ensemble of two models to improve performance further.

```bash
python train_ensemble_smart.py --model-a densenet121 --model-b efficientnet_v2_s
```

## 📂 Output

Training results, including metrics (Accuracy, F1-Score, Recall), confusion matrices, and loss curves, are saved in the `outputs/` directory. Each run generates:
*   `kfold_results.json`: Detailed metrics for every fold.
*   `*.png`: plots for loss and accuracy.
*   `*.md` and `*.pdf`: A summary report of the training run.

## 📈 Results Overview

Preliminary results show strong performance across all three architectures, with individual models achieving **>98% accuracy** on validation folds.
