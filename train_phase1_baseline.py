import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torchvision import datasets, models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=20),
                transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
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


def build_model(name: str) -> nn.Module:
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
        return model
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    raise ValueError(f"Unsupported model: {name}")


def build_weighted_sampler(subset: Subset) -> WeightedRandomSampler:
    counts = [0, 0]
    targets = subset.dataset.targets
    for idx in subset.indices:
        counts[targets[idx]] += 1
    total = sum(counts)
    class_weights = [total / (2 * c) if c > 0 else 0.0 for c in counts]
    sample_weights = [class_weights[targets[idx]] for idx in subset.indices]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def compute_pos_weight(subset: Subset) -> torch.Tensor:
    counts = [0, 0]
    targets = subset.dataset.targets
    for idx in subset.indices:
        counts[targets[idx]] += 1
    neg, pos = counts[0], counts[1]
    pos_weight = neg / pos if pos > 0 else 1.0
    return torch.tensor([pos_weight], dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train: bool,
) -> Tuple[float, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).float().unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, targets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()
        correct += (preds == targets.long()).sum().item()
        total += targets.numel()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 baseline: EfficientNet-B0 binary classifier")
    parser.add_argument("--train-dir", default="DATASET/train/train", help="Path to training data root")
    parser.add_argument("--test-dir", default="DATASET/test/test", help="Path to test data root")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu")
    parser.add_argument("--model", default="efficientnet_b0", help="Backbone (efficientnet_b0, resnet50)")
    parser.add_argument("--aug-preset", default="basic", choices=["basic", "strong"], help="Augmentation strength")
    parser.add_argument("--sampler", default="shuffle", choices=["shuffle", "weighted"], help="Sampling strategy")
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=-1.0,
        help="Positive class weight for BCE loss. Use -1 to auto-compute from train split.",
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

    sampler = None
    if args.sampler == "weighted":
        sampler = build_weighted_sampler(train_subset)

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

    model = build_model(args.model).to(device)
    if args.pos_weight >= 0:
        pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32)
    else:
        pos_weight = compute_pos_weight(train_subset)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_path = os.path.join(args.output_dir, "efficientnet_b0_phase1_best.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    with torch.no_grad():
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)

    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
