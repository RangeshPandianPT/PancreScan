# PancreScan K-Fold Cross-Validation Report

Date: 2026-02-10

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
| 1 | 0.049943 | 0.9820 | 0.9795 | 0.9844 | 0.9817 | 1.0000 | 0.9689 | 0.9842 |
| 2 | 0.037007 | 0.9800 | 0.9794 | 0.9794 | 0.9794 | 0.9827 | 0.9827 | 0.9827 |

## Summary (Mean ± Std)

| Metric | Mean | Std |
| --- | --- | --- |
| loss | 0.043475 | 0.006468 |
| accuracy | 0.980980 | 0.001020 |
| precision | 0.979495 | 0.000050 |
| recall | 0.981937 | 0.002492 |
| f1 | 0.980548 | 0.001103 |
| pos_precision | 0.991349 | 0.008651 |
| pos_recall | 0.975779 | 0.006920 |
| pos_f1 | 0.983441 | 0.000742 |

## Notes
- Metrics are computed on the validation split of each fold.
- Positive-class metrics use the selected threshold.
