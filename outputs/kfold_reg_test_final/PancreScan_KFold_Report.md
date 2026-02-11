# PancreScan K-Fold Cross-Validation Report

Date: 2026-02-10

## Regularization Settings
- Early Stopping Patience: 2
- Dropout: 0.5
- Backbone Frozen: True
- Weight Decay: 0.0001

## Configuration
- Data directory: DATASET/train/train
- Model: efficientnet_b0
- Folds: 2
- Epochs per fold: 1
- Batch size: 16
- Image size: 224
- Learning rate: 0.0003
- Augmentation preset: basic
- Sampler: shuffle
- Positive class: pancreatic_tumor
- Positive threshold: 0.4
- Classes: normal, pancreatic_tumor

## Fold Metrics (Validation)

| Fold | Loss | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Precision (pos) | Recall (pos) | F1 (pos) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.348295 | 0.9060 | 0.9197 | 0.8925 | 0.9009 | 0.8735 | 0.9792 | 0.9233 |
| 2 | 0.274880 | 0.9840 | 0.9817 | 0.9862 | 0.9836 | 1.0000 | 0.9723 | 0.9860 |

## Summary (Mean ± Std)

| Metric | Mean | Std |
| --- | --- | --- |
| loss | 0.311587 | 0.036708 |
| accuracy | 0.944984 | 0.038984 |
| precision | 0.950667 | 0.030984 |
| recall | 0.939311 | 0.046848 |
| f1 | 0.942289 | 0.041348 |
| pos_precision | 0.936728 | 0.063272 |
| pos_recall | 0.975779 | 0.003460 |
| pos_f1 | 0.954646 | 0.031319 |

## Notes
- Metrics are computed on the validation split of each fold.
- Positive-class metrics use the selected threshold.
