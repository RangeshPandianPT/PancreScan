import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms


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


def split_dataset(
    dataset: datasets.ImageFolder,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    return train_subset, val_subset


def compute_class_weights(subset: Subset, num_classes: int) -> torch.Tensor:
    counts = [0] * num_classes
    targets = subset.dataset.targets
    for idx in subset.indices:
        counts[targets[idx]] += 1
    total = sum(counts)
    weights = [total / (num_classes * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(subset: Subset, num_classes: int) -> WeightedRandomSampler:
    class_weights = compute_class_weights(subset, num_classes)
    targets = subset.dataset.targets
    sample_weights = [class_weights[targets[idx]].item() for idx in subset.indices]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {name}")


class FocalLoss(nn.Module):
    def __init__(self, gamma: float, weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1.0 - pt) ** self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight.gather(0, targets)
        return loss.mean()


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


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    output_dir: str,
    class_weights: torch.Tensor,
    positive_class: int,
    threshold: float,
    loss_name: str,
    focal_gamma: float,
) -> str:
    model = build_model(model_name, num_classes).to(device)
    if loss_name == "focal":
        criterion = FocalLoss(gamma=focal_gamma, weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    best_path = os.path.join(output_dir, f"{model_name}_best.pt")

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
            f"[{model_name}] Epoch {epoch}/{epochs} "
            f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} "
            f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1:.4f} "
            f"val_recall_pos={val_metrics.pos_recall:.4f}"
        )

        if val_metrics.loss < best_val_loss:
            best_val_loss = val_metrics.loss
            torch.save(model.state_dict(), best_path)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best_path


def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    positive_class: int,
    threshold: float,
    loss_name: str,
    focal_gamma: float,
) -> Metrics:
    model = build_model(model_name, num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if loss_name == "focal":
        criterion = FocalLoss(gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    with torch.no_grad():
        metrics = run_epoch(
            model,
            loader,
            criterion,
            None,
            scaler,
            device,
            train=False,
            positive_class=positive_class,
            threshold=threshold,
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


def evaluate_ensemble(
    model_a_name: str,
    model_b_name: str,
    model_a_ckpt: str,
    model_b_ckpt: str,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    positive_class: int,
    threshold: float,
    weights: Tuple[float, float],
) -> Metrics:
    model_a = build_model(model_a_name, num_classes).to(device)
    model_b = build_model(model_b_name, num_classes).to(device)
    model_a.load_state_dict(torch.load(model_a_ckpt, map_location=device))
    model_b.load_state_dict(torch.load(model_b_ckpt, map_location=device))
    model_a.eval()
    model_b.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            logits_a = model_a(images)
            logits_b = model_b(images)
            logits = logits_a * weights[0] + logits_b * weights[1]
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits)
            all_targets.append(targets)

    total_loss /= len(loader.dataset)
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    del model_a
    del model_b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return compute_metrics(logits, targets, total_loss, positive_class, threshold)


def parse_ensemble_weights(raw: str) -> Tuple[float, float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--ensemble-weights must be two comma-separated values, e.g. 0.6,0.4")
    weights = [float(parts[0]), float(parts[1])]
    total = weights[0] + weights[1]
    if total <= 0:
        raise ValueError("--ensemble-weights must sum to a positive value")
    return weights[0] / total, weights[1] / total


def save_run_config(path: str, config: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Mode ensemble training for RTX 3050.")
    parser.add_argument("--train-dir", default="DATASET/train/train", help="Path to training data root.")
    parser.add_argument("--test-dir", default="DATASET/test/test", help="Path to test data root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs per model.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", default="outputs", help="Output directory.")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu.")
    parser.add_argument("--model-a", default="densenet121", help="First backbone (densenet121, efficientnet_b0, resnet50, efficientnet_v2_s, convnext_tiny).")
    parser.add_argument("--model-b", default="efficientnet_b0", help="Second backbone (densenet121, efficientnet_b0, resnet50, efficientnet_v2_s, convnext_tiny).")
    parser.add_argument("--aug-preset", default="basic", choices=["basic", "strong"], help="Augmentation strength.")
    parser.add_argument("--sampler", default="shuffle", choices=["shuffle", "weighted"], help="Sampling strategy.")
    parser.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "focal"], help="Loss function.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
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
        "--ensemble-weights",
        default="0.5,0.5",
        help="Comma-separated weights for DenseNet121,EfficientNet-B0.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    os.makedirs(args.output_dir, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.image_size, args.aug_preset)

    full_train = datasets.ImageFolder(args.train_dir, transform=train_tf)
    train_subset, val_subset = split_dataset(full_train, args.val_ratio, args.seed)

    val_dataset = datasets.ImageFolder(args.train_dir, transform=eval_tf)
    val_subset = Subset(val_dataset, val_subset.indices)

    test_dataset = datasets.ImageFolder(args.test_dir, transform=eval_tf)

    num_classes = len(full_train.classes)
    positive_class = (
        full_train.classes.index(args.positive_class)
        if args.positive_class in full_train.classes
        else 1
    )
    ensemble_weights = parse_ensemble_weights(args.ensemble_weights)
    class_weights = compute_class_weights(train_subset, num_classes)
    class_counts = [0] * num_classes
    for idx in train_subset.indices:
        class_counts[full_train.targets[idx]] += 1
    print(f"Train class counts: {class_counts}")

    sampler = None
    if args.sampler == "weighted":
        sampler = build_weighted_sampler(train_subset, num_classes)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = vars(args)
    config["classes"] = full_train.classes
    config["positive_class"] = positive_class
    config["ensemble_weights"] = ensemble_weights
    save_run_config(os.path.join(args.output_dir, "run_config.json"), config)

    print("Training DenseNet121...")
    densenet_path = train_model(
        args.model_a,
        train_loader,
        val_loader,
        num_classes,
        device,
        args.epochs,
        args.lr,
        args.output_dir,
        class_weights,
        positive_class,
        args.pos_threshold,
        args.loss,
        args.focal_gamma,
    )

    print("Training EfficientNet-B0...")
    efficientnet_path = train_model(
        args.model_b,
        train_loader,
        val_loader,
        num_classes,
        device,
        args.epochs,
        args.lr,
        args.output_dir,
        class_weights,
        positive_class,
        args.pos_threshold,
        args.loss,
        args.focal_gamma,
    )

    print("Evaluating on test set...")
    dense_metrics = evaluate_model(
        args.model_a,
        densenet_path,
        test_loader,
        num_classes,
        device,
        positive_class,
        args.pos_threshold,
        args.loss,
        args.focal_gamma,
    )
    eff_metrics = evaluate_model(
        args.model_b,
        efficientnet_path,
        test_loader,
        num_classes,
        device,
        positive_class,
        args.pos_threshold,
        args.loss,
        args.focal_gamma,
    )
    ens_metrics = evaluate_ensemble(
        args.model_a,
        args.model_b,
        densenet_path,
        efficientnet_path,
        test_loader,
        num_classes,
        device,
        positive_class,
        args.pos_threshold,
        ensemble_weights,
    )

    results = {
        "densenet121": dense_metrics.__dict__,
        "efficientnet_b0": eff_metrics.__dict__,
        "ensemble": ens_metrics.__dict__,
    }

    results_path = os.path.join(args.output_dir, "test_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
