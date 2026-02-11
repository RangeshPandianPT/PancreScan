# PancreScan K-Fold Cross-Validation Report

Date: 2026-02-11

## Regularization Settings
- Early Stopping Patience: 3
- Dropout: 0.5
- Backbone Frozen: True
- Weight Decay: 0.0001

## Configuration
- Data directory: DATASET/train/train
- Model: convnext_tiny
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
| 1 | 0.043276 | 0.9751 | 0.9739 | 0.9753 | 0.9746 | 0.9826 | 0.9741 | 0.9784 |
| 2 | 0.040556 | 0.9950 | 0.9941 | 0.9957 | 0.9949 | 1.0000 | 0.9914 | 0.9957 |
| 3 | 0.068797 | 0.9800 | 0.9773 | 0.9828 | 0.9796 | 1.0000 | 0.9655 | 0.9825 |
| 4 | 0.042428 | 0.9849 | 0.9828 | 0.9870 | 0.9846 | 1.0000 | 0.9739 | 0.9868 |
| 5 | 0.040624 | 0.9799 | 0.9794 | 0.9794 | 0.9794 | 0.9826 | 0.9826 | 0.9826 |

## Summary (Mean ± Std)

| Metric | Mean | Std |
| --- | --- | --- |
| loss | 0.047136 | 0.010881 |
| accuracy | 0.982990 | 0.006758 |
| precision | 0.981482 | 0.006948 |
| recall | 0.984022 | 0.006982 |
| f1 | 0.982611 | 0.006910 |
| pos_precision | 0.993043 | 0.008520 |
| pos_recall | 0.977511 | 0.008792 |
| pos_f1 | 0.985175 | 0.005886 |

## Notes
- Metrics are computed on the validation split of each fold.
- Positive-class metrics use the selected threshold.
