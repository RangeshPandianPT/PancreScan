import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

# Adjust path to import models
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.unet import UNetMultiTask

class PancreasDataset(Dataset):
    def __init__(self, image_dir, mask_dir, split="train", val_split=0.2, seed=42, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        files = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        files.sort()
        
        train_files, val_files = train_test_split(files, test_size=val_split, random_state=seed)
        self.files = train_files if split == "train" else val_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # 1 Channel
        
        # Determine classification label from mask (if any pixel > 0 -> Tumor classes as 1)
        mask_np = np.array(mask)
        label = 1.0 if np.sum(mask_np) > 0 else 0.0
        
        if self.transform:
            # We must apply the same random crop/resize to both. For simplicity here:
            image = image.resize((224, 224), Image.BILINEAR)
            mask = mask.resize((224, 224), Image.NEAREST)
            
            image_tensor = transforms.ToTensor()(image)
            image_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image_tensor)
            
            mask_tensor = transforms.ToTensor()(mask)
            mask_tensor = (mask_tensor > 0.1).float() # Binarize
        else:
            image_tensor = transforms.ToTensor()(image)
            mask_tensor = transforms.ToTensor()(mask)
            
        return image_tensor, mask_tensor, torch.tensor([label], dtype=torch.float32)

def train_unet():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="outputs/unet")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = "data/images"
    masks_dir = "data/masks"

    print("Loading data...")
    train_dataset = PancreasDataset(images_dir, masks_dir, split="train", transform=True)
    val_dataset = PancreasDataset(images_dir, masks_dir, split="val", transform=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNetMultiTask(n_channels=3, n_classes=1, num_cls_classes=1)
    model.to(args.device)

    # Use BCEWithLogitsLoss for both Segmentation and Classification
    criterion_seg = nn.BCEWithLogitsLoss()
    criterion_cls = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    print(f"Starting Multi-Task U-Net training on {args.device}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        
        for images, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, masks, labels = images.to(args.device), masks.to(args.device), labels.to(args.device)
            
            optimizer.zero_grad()
            seg_logits, cls_logits = model(images)
            
            loss_seg = criterion_seg(seg_logits, masks)
            loss_cls = criterion_cls(cls_logits, labels)
            
            # Combine losses
            loss = loss_seg + loss_cls
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_seg_loss += loss_seg.item()
            train_cls_loss += loss_cls.item()

        model.eval()
        val_loss = 0.0
        correct_cls = 0
        total_cls = 0
        
        with torch.no_grad():
            for images, masks, labels in val_loader:
                images, masks, labels = images.to(args.device), masks.to(args.device), labels.to(args.device)
                
                seg_logits, cls_logits = model(images)
                
                loss_seg = criterion_seg(seg_logits, masks)
                loss_cls = criterion_cls(cls_logits, labels)
                val_loss += (loss_seg.item() + loss_cls.item())
                
                preds = torch.sigmoid(cls_logits) >= 0.5
                correct_cls += (preds == labels).sum().item()
                total_cls += labels.size(0)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc = correct_cls / total_cls

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} (Seg: {train_seg_loss/len(train_loader):.4f}, Cls: {train_cls_loss/len(train_loader):.4f}) | Val Loss: {val_loss:.4f} | Val Cls Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, "unet_multitask_best.pt"))
            print(f">>> Logged Best Model (Val Loss: {val_loss:.4f})")

if __name__ == "__main__":
    train_unet()
