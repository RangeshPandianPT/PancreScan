import argparse
import json
import os
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Tuple

import markdown
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms
from xhtml2pdf import pisa


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    pos_precision: float
    pos_recall: float
    pos_f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms(image_size: int, preset: str) -> Tuple[transforms.Compose, transforms.Compose]:
    if preset == "strong":
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        return model
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        return model
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )
        return model
    raise ValueError(f"Unsupported model: {name}")


def compute_class_weights(indices: List[int], targets: List[int], num_classes: int) -> torch.Tensor:
    counts = [0] * num_classes
    for idx in indices:
        counts[targets[idx]] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(indices: List[int], targets: List[int], num_classes: int) -> WeightedRandomSampler:
    class_weights = compute_class_weights(indices, targets, num_classes)
    sample_weights = [class_weights[targets[idx]].item() for idx in indices]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def stratified_kfold(indices_by_class: List[List[int]], k_folds: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    folds = [[] for _ in range(k_folds)]
    for cls_indices in indices_by_class:
        shuffled = cls_indices[:]
        rng.shuffle(shuffled)
        chunks = [shuffled[i::k_folds] for i in range(k_folds)]
        for fold_id, chunk in enumerate(chunks):
            folds[fold_id].extend(chunk)
    return folds


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss: float,
    positive_class: int,
    threshold: float,
) -> Metrics:
    num_classes = logits.size(1)
    probs = torch.softmax(logits, dim=1)
    if num_classes == 2:
        pos_probs = probs[:, positive_class]
        preds = torch.where(pos_probs >= threshold, positive_class, 1 - positive_class)
    else:
        preds = torch.argmax(logits, dim=1)

    correct = (preds == targets).sum().item()
    total = targets.numel()
    accuracy = correct / total if total > 0 else 0.0

    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().item()
        fp = ((preds == cls) & (targets != cls)).sum().item()
        fn = ((preds != cls) & (targets == cls)).sum().item()
        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cls_f1 = (
            2 * cls_precision * cls_recall / (cls_precision + cls_recall)
            if (cls_precision + cls_recall) > 0
            else 0.0
        )
        precision += cls_precision
        recall += cls_recall
        f1 += cls_f1

    precision /= num_classes
    recall /= num_classes
    f1 /= num_classes

    tp_pos = ((preds == positive_class) & (targets == positive_class)).sum().item()
    fp_pos = ((preds == positive_class) & (targets != positive_class)).sum().item()
    fn_pos = ((preds != positive_class) & (targets == positive_class)).sum().item()
    pos_precision = tp_pos / (tp_pos + fp_pos) if (tp_pos + fp_pos) > 0 else 0.0
    pos_recall = tp_pos / (tp_pos + fn_pos) if (tp_pos + fn_pos) > 0 else 0.0
    pos_f1 = (
        2 * pos_precision * pos_recall / (pos_precision + pos_recall)
        if (pos_precision + pos_recall) > 0
        else 0.0
    )

    return Metrics(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        pos_precision=pos_precision,
        pos_recall=pos_recall,
        pos_f1=pos_f1,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    train: bool,
    positive_class: int,
    threshold: float,
) -> Metrics:
    model.train() if train else model.eval()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach())
        all_targets.append(targets.detach())

    total_loss /= len(loader.dataset)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_metrics(logits, targets, total_loss, positive_class, threshold)


def train_one_fold(
    fold_id: int,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    class_weights: torch.Tensor,
    positive_class: int,
    threshold: float,
    patience: int,
    freeze_backbone: bool,
    weight_decay: float,
    output_dir: str = "outputs",
) -> Metrics:
    save_dir = os.path.join(output_dir, f"kfold_{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    model = build_model(model_name, num_classes)
    
    if freeze_backbone:
        for name, param in model.named_parameters():
             if "classifier" not in name and "fc" not in name:
                 param.requires_grad = False

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_loss = float("inf")
    best_metrics = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            train=True,
            positive_class=positive_class,
            threshold=threshold,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer,
                scaler,
                device,
                train=False,
                positive_class=positive_class,
                threshold=threshold,
            )

        scheduler.step(val_metrics.loss)
        print(
            f"Fold {fold_id} Epoch {epoch}/{epochs} "
            f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f} "
            f"val_recall_pos={val_metrics.pos_recall:.4f}"
        )

        if val_metrics.loss < best_loss:
            best_loss = val_metrics.loss
            best_metrics = val_metrics
            best_loss = val_metrics.loss
            best_metrics = val_metrics
            patience_counter = 0
            # Save best model
            # Save best model
            save_path = f"{model_name}_fold_{fold_id}_best.pt"
            if output_dir:
                save_path = os.path.join(output_dir, save_path)
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if best_metrics is None:
        raise RuntimeError("No validation metrics captured for fold")

    return best_metrics


def summarize_metrics(metrics_list: List[Metrics]) -> Dict[str, Dict[str, float]]:
    def mean_std(values: List[float]) -> Dict[str, float]:
        mean = statistics.mean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std}

    return {
        "loss": mean_std([m.loss for m in metrics_list]),
        "accuracy": mean_std([m.accuracy for m in metrics_list]),
        "precision": mean_std([m.precision for m in metrics_list]),
        "recall": mean_std([m.recall for m in metrics_list]),
        "f1": mean_std([m.f1 for m in metrics_list]),
        "pos_precision": mean_std([m.pos_precision for m in metrics_list]),
        "pos_recall": mean_std([m.pos_recall for m in metrics_list]),
        "pos_f1": mean_std([m.pos_f1 for m in metrics_list]),
    }


def build_report_markdown(
    config: Dict[str, object],
    classes: List[str],
    fold_metrics: List[Metrics],
    summary: Dict[str, Dict[str, float]],
) -> str:
    lines = []
    lines.append("# PancreScan K-Fold Cross-Validation Report")
    lines.append("")
    lines.append(f"Date: {config['report_date']}")
    lines.append("")
    lines.append("## Regularization Settings")
    lines.append(f"- Early Stopping Patience: {config.get('patience', 'N/A')}")
    lines.append(f"- Dropout: {config.get('dropout', 'N/A')}")
    lines.append(f"- Backbone Frozen: {config.get('freeze_backbone', 'False')}")
    lines.append(f"- Weight Decay: {config.get('weight_decay', 'N/A')}")
    lines.append("")
    lines.append("## Configuration")
    lines.append(f"- Data directory: {config['data_dir']}")
    lines.append(f"- Model: {config['model']}")
    lines.append(f"- Folds: {config['k_folds']}")
    lines.append(f"- Epochs per fold: {config['epochs']}")
    lines.append(f"- Batch size: {config['batch_size']}")
    lines.append(f"- Image size: {config['image_size']}")
    lines.append(f"- Learning rate: {config['lr']}")
    lines.append(f"- Augmentation preset: {config['aug_preset']}")
    lines.append(f"- Sampler: {config['sampler']}")
    lines.append(f"- Positive class: {config['positive_class_name']}")
    lines.append(f"- Positive threshold: {config['pos_threshold']}")
    lines.append(f"- Classes: {', '.join(classes)}")
    lines.append("")
    lines.append("## Fold Metrics (Validation)")
    lines.append("")
    lines.append(
        "| Fold | Loss | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Precision (pos) | Recall (pos) | F1 (pos) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for idx, metrics in enumerate(fold_metrics, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"{metrics.loss:.6f}",
                    f"{metrics.accuracy:.4f}",
                    f"{metrics.precision:.4f}",
                    f"{metrics.recall:.4f}",
                    f"{metrics.f1:.4f}",
                    f"{metrics.pos_precision:.4f}",
                    f"{metrics.pos_recall:.4f}",
                    f"{metrics.pos_f1:.4f}",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Summary (Mean ± Std)")
    lines.append("")
    lines.append(
        "| Metric | Mean | Std |\n| --- | --- | --- |"
    )
    for metric_name, stats in summary.items():
        lines.append(
            f"| {metric_name} | {stats['mean']:.6f} | {stats['std']:.6f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Metrics are computed on the validation split of each fold.")
    lines.append("- Positive-class metrics use the selected threshold.")
    lines.append("")
    return "\n".join(lines)


def render_pdf(markdown_text: str, output_pdf_path: str) -> None:
    html_text = markdown.markdown(markdown_text, extensions=["tables"])
    html_content = f"""
<html>
<head>
<style>
    body {{
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 11pt;
        line-height: 1.6;
        color: #2c3e50;
    }}
    h1 {{
        color: #2c3e50;
        border-bottom: 2px solid #2c3e50;
        padding-bottom: 10px;
        font-size: 22pt;
        margin-top: 0;
    }}
    h2 {{
        color: #34495e;
        margin-top: 22px;
        font-size: 16pt;
        border-bottom: 1px solid #eee;
        padding-bottom: 5px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 16px 0;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #BDC3C7;
        padding: 8px;
        text-align: left;
    }}
    th {{
        background-color: #ECF0F1;
        color: #2c3e50;
        font-weight: bold;
    }}
    tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    code {{
        background-color: #f4f4f4;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', Courier, monospace;
        font-size: 10pt;
        color: #c0392b;
    }}
</style>
</head>
<body>
{html_text}
</body>
</html>
"""

    with open(output_pdf_path, "wb") as pdf_file:
        status = pisa.CreatePDF(html_content, dest=pdf_file)

    if status.err:
        raise RuntimeError(f"Error creating PDF: {status.err}")


def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold cross-validation with PDF report.")
    parser.add_argument("--data-dir", default="DATASET/train/train", help="Path to data root.")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu.")
    parser.add_argument("--model", default="efficientnet_b0", help="Backbone (densenet121, efficientnet_b0, resnet50, efficientnet_v2_s, convnext_tiny).")
    parser.add_argument("--aug-preset", default="basic", choices=["basic", "strong"], help="Augmentation strength.")
    parser.add_argument("--sampler", default="shuffle", choices=["shuffle", "weighted"], help="Sampling strategy.")
    parser.add_argument(
        "--positive-class",
        default="pancreatic_tumor",
        help="Class name to treat as positive (used for recall focus).",
    )
    parser.add_argument(
        "--pos-threshold",
        type=float,
        default=0.4,
        help="Probability threshold for positive class (lower boosts recall).",
    )
    parser.add_argument(
        "--report-name",
        default="PancreScan_KFold_Report",
        help="Base name for report files (without extension).",
    )
    parser.add_argument(
        "--report-date",
        default="2026-02-10",
        help="Report date string.",
    )
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability (hardcoded to 0.5 in model build for now).")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone layers.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)

    os.makedirs(args.output_dir, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.image_size, args.aug_preset)

    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_tf)
    num_classes = len(full_dataset.classes)
    positive_class = (
        full_dataset.classes.index(args.positive_class)
        if args.positive_class in full_dataset.classes
        else 1
    )

    indices_by_class = [[] for _ in range(num_classes)]
    for idx, target in enumerate(full_dataset.targets):
        indices_by_class[target].append(idx)

    fold_indices = stratified_kfold(indices_by_class, args.k_folds, args.seed)
    fold_metrics = []

    for fold_id, val_indices in enumerate(fold_indices, start=1):
        train_indices = []
        for other_fold_id, fold in enumerate(fold_indices, start=1):
            if other_fold_id != fold_id:
                train_indices.extend(fold)

        train_subset = Subset(full_dataset, train_indices)
        val_dataset = datasets.ImageFolder(args.data_dir, transform=eval_tf)
        val_subset = Subset(val_dataset, val_indices)

        class_weights = compute_class_weights(train_indices, full_dataset.targets, num_classes)
        class_counts = [0] * num_classes
        for idx in train_indices:
            class_counts[full_dataset.targets[idx]] += 1
        print(f"Fold {fold_id} train class counts: {class_counts}")

        sampler = None
        if args.sampler == "weighted":
            sampler = build_weighted_sampler(train_indices, full_dataset.targets, num_classes)

        train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        metrics = train_one_fold(
            fold_id,
            args.model,
            train_loader,
            val_loader,
            num_classes,
            device,
            args.epochs,
            args.lr,
            class_weights,
            positive_class,
            args.pos_threshold,
            args.patience,
            args.freeze_backbone,
            args.weight_decay,
            output_dir=args.output_dir,
        )
        fold_metrics.append(metrics)

    summary = summarize_metrics(fold_metrics)

    results = {
        "config": vars(args),
        "classes": full_dataset.classes,
        "positive_class_index": positive_class,
        "fold_metrics": [m.__dict__ for m in fold_metrics],
        "summary": summary,
    }

    results_path = os.path.join(args.output_dir, "kfold_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    report_config = {
        "data_dir": args.data_dir,
        "model": args.model,
        "k_folds": args.k_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "lr": args.lr,
        "aug_preset": args.aug_preset,
        "sampler": args.sampler,
        "positive_class_name": args.positive_class,
        "pos_threshold": args.pos_threshold,
        "report_date": args.report_date,
        "patience": args.patience,
        "dropout": args.dropout,
        "freeze_backbone": args.freeze_backbone,
        "weight_decay": args.weight_decay,
    }
    report_md = build_report_markdown(report_config, full_dataset.classes, fold_metrics, summary)

    report_md_path = os.path.join(args.output_dir, f"{args.report_name}.md")
    report_pdf_path = os.path.join(args.output_dir, f"{args.report_name}.pdf")

    with open(report_md_path, "w", encoding="utf-8") as handle:
        handle.write(report_md)

    render_pdf(report_md, report_pdf_path)

    print(f"Saved results JSON: {os.path.abspath(results_path)}")
    print(f"Saved report: {os.path.abspath(report_md_path)}")
    print(f"Saved PDF: {os.path.abspath(report_pdf_path)}")


if __name__ == "__main__":
    main()
