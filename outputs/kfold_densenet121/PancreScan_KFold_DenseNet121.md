# PancreScan K-Fold Cross-Validation Report

Date: 2026-02-11

## Regularization Settings
- Early Stopping Patience: 3
- Dropout: 0.5
- Backbone Frozen: True
- Weight Decay: 0.0001

## Configuration
- Data directory: DATASET/train/train
- Model: densenet121
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
| 1 | 0.109749 | 0.9801 | 0.9775 | 0.9828 | 0.9797 | 1.0000 | 0.9655 | 0.9825 |
| 2 | 0.063302 | 0.9900 | 0.9884 | 0.9914 | 0.9898 | 1.0000 | 0.9828 | 0.9913 |
| 3 | 0.118588 | 0.9650 | 0.9615 | 0.9698 | 0.9644 | 1.0000 | 0.9397 | 0.9689 |
| 4 | 0.070078 | 0.9849 | 0.9828 | 0.9870 | 0.9846 | 1.0000 | 0.9739 | 0.9868 |
| 5 | 0.052097 | 0.9899 | 0.9884 | 0.9913 | 0.9897 | 1.0000 | 0.9826 | 0.9912 |

## Summary (Mean ± Std)

| Metric | Mean | Std |
| --- | --- | --- |
| loss | 0.082763 | 0.026426 |
| accuracy | 0.981995 | 0.009256 |
| precision | 0.979714 | 0.009943 |
| recall | 0.984445 | 0.007974 |
| f1 | 0.981660 | 0.009378 |
| pos_precision | 1.000000 | 0.000000 |
| pos_recall | 0.968891 | 0.015949 |
| pos_f1 | 0.984132 | 0.008294 |

## Notes
- Metrics are computed on the validation split of each fold.
- Positive-class metrics use the selected threshold.
