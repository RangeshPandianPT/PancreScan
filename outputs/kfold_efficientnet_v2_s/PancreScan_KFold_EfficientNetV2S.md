# PancreScan K-Fold Cross-Validation Report

Date: 2026-02-11

## Regularization Settings
- Early Stopping Patience: 3
- Dropout: 0.5
- Backbone Frozen: True
- Weight Decay: 0.0001

## Configuration
- Data directory: DATASET/train/train
- Model: efficientnet_v2_s
- Folds: 5
- Epochs per fold: 20
- Batch size: 16
- Image size: 224
- Learning rate: 0.0003
- Augmentation preset: strong
- Sampler: weighted
- Positive class: pancreatic_tumor
- Positive threshold: 0.4
- Classes: normal, pancreatic_tumor

## Fold Metrics (Validation)

| Fold | Loss | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Precision (pos) | Recall (pos) | F1 (pos) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.050218 | 0.9751 | 0.9739 | 0.9753 | 0.9746 | 0.9826 | 0.9741 | 0.9784 |
| 2 | 0.039860 | 0.9850 | 0.9854 | 0.9838 | 0.9846 | 0.9829 | 0.9914 | 0.9871 |
| 3 | 0.036972 | 0.9950 | 0.9941 | 0.9957 | 0.9949 | 1.0000 | 0.9914 | 0.9957 |
| 4 | 0.042479 | 0.9698 | 0.9727 | 0.9659 | 0.9689 | 0.9580 | 0.9913 | 0.9744 |
| 5 | 0.016071 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Summary (Mean ± Std)

| Metric | Mean | Std |
| --- | --- | --- |
| loss | 0.037120 | 0.011409 |
| accuracy | 0.984995 | 0.011420 |
| precision | 0.985230 | 0.010792 |
| recall | 0.984134 | 0.012619 |
| f1 | 0.984580 | 0.011752 |
| pos_precision | 0.984700 | 0.015424 |
| pos_recall | 0.989640 | 0.008444 |
| pos_f1 | 0.987102 | 0.009784 |

## Notes
- Metrics are computed on the validation split of each fold.
- Positive-class metrics use the selected threshold.
