# PancreScan Training Report

Date: 2026-02-09

## Overview
This report summarizes the PancreScan training run using an ensemble of DenseNet121 and EfficientNet-B0 for binary classification of CT slices into Normal vs Tumor.

## Dataset
- Train directory: DATASET/train/train
- Test directory: DATASET/test/test
- Class labels: normal, pancreatic_tumor
- Validation split: 0.20

## Training Configuration
- Image size: 224 x 224
- Batch size: 16
- Epochs: 20
- Learning rate: 0.0003
- Optimizer: AdamW
- Seed: 42
- Device setting: auto
- Positive class index: 1 (pancreatic_tumor)
- Positive threshold: 0.40
- Ensemble weights (DenseNet121, EfficientNet-B0): 0.50, 0.50

## Models
- DenseNet121 (ImageNet pre-trained, fine-tuned)
- EfficientNet-B0 (ImageNet pre-trained, fine-tuned)
- Ensemble: weighted average of logits

## Test Results
All metrics are computed on the test set.

| Model | Loss | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Precision (tumor) | Recall (tumor) | F1 (tumor) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DenseNet121 | 0.000724 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| EfficientNet-B0 | 0.001321 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Ensemble | 0.000744 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Notes
- Tumor recall is prioritized by using a positive threshold of 0.40.
- Checkpoints saved in outputs/ (efficientnet_b0_best.pt, densenet121_best.pt).
- Report generated from outputs/run_config.json and outputs/test_results.json.
