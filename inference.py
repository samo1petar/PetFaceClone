#!/usr/bin/env python3
"""
Single-GPU training script for pet face verification.

Trains a backbone to produce L2-normalized embeddings where same-animal
vectors are close and different-animal vectors are far apart.

Usage:
    python train.py --epochs 20 --network r50 --loss arcface
    python train.py --epochs 1 --batch-size 16  # quick sanity check
"""

import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add src/ to path so we can import backbones
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from backbones import get_model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DogDataset(Dataset):
    """Directory-based dataset: data_root/[id]/*.{png,jpg,jpeg}"""

    def __init__(self, root, identity_ids, transform=None):
        """
        Args:
            root: path like dataset/original/dog
            identity_ids: list of subdirectory names to include
            transform: torchvision transforms
        """
        self.transform = transform
        self.samples = []  # (path, label)
        self.label_to_indices = defaultdict(list)

        for label, id_name in enumerate(sorted(identity_ids)):
            id_dir = os.path.join(root, id_name)
            for fname in sorted(os.listdir(id_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    idx = len(self.samples)
                    self.samples.append((os.path.join(id_dir, fname), label))
                    self.label_to_indices[label].append(idx)

        self.num_classes = len(identity_ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class ArcFaceLoss(nn.Module):
    """ArcFace (Additive Angular Margin) loss."""

    def __init__(self, embedding_size, num_classes, margin=0.5, scale=64.0):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.margin = margin
        self.scale = scale
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        weight = F.normalize(self.weight)

        cosine = F.linear(embeddings, weight)
        sine = torch.sqrt(1.0 - torch.clamp(cosine * cosine, 0, 1))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.unsqueeze(1), 1)

        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale
        return self.criterion(logits, labels)


class TripletLoss(nn.Module):
    """Triplet loss with online batch-hard mining."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        dist = torch.cdist(embeddings, embeddings, p=2)

        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Hardest positive: max distance among same-class pairs (exclude self)
        mask_pos = labels_eq.clone()
        mask_pos.fill_diagonal_(False)
        dist_pos = dist * mask_pos.float()
        hardest_pos, _ = dist_pos.max(dim=1)

        # Hardest negative: min distance among different-class pairs
        dist_neg = dist + labels_eq.float() * 1e6
        hardest_neg, _ = dist_neg.min(dim=1)

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()


class ContrastiveLoss(nn.Module):
    """Contrastive loss on consecutive pairs within the batch."""

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings)
        n = embeddings.size(0)

        idx1 = torch.arange(0, n - 1, device=embeddings.device)
        idx2 = torch.arange(1, n, device=embeddings.device)

        dist = F.pairwise_distance(embeddings[idx1], embeddings[idx2])
        same = (labels[idx1] == labels[idx2]).float()

        loss = same * dist.pow(2) + (1 - same) * F.relu(self.margin - dist).pow(2)
        return loss.mean()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(backbone, val_dataset, device, num_pairs=5000):
    """Compute verification AUC and EER on held-out identities."""
    backbone.eval()

    loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True,
    )
    all_embeddings = []
    all_labels = []

    for imgs, labels in loader:
        emb = F.normalize(backbone(imgs.to(device)))
        all_embeddings.append(emb.cpu())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    # Build label -> indices map, keep only labels with >= 2 samples
    label_to_idx = defaultdict(list)
    for i, lab in enumerate(all_labels.tolist()):
        label_to_idx[lab].append(i)

    valid_labels = [lab for lab, idxs in label_to_idx.items() if len(idxs) >= 2]
    all_label_list = list(label_to_idx.keys())

    if len(valid_labels) < 2:
        return None, None

    sims, targets = [], []
    for _ in range(num_pairs // 2):
        # Positive pair
        lab = random.choice(valid_labels)
        i, j = random.sample(label_to_idx[lab], 2)
        sim = F.cosine_similarity(
            all_embeddings[i : i + 1], all_embeddings[j : j + 1],
        ).item()
        sims.append(sim)
        targets.append(1)

        # Negative pair
        lab1, lab2 = random.sample(all_label_list, 2)
        i = random.choice(label_to_idx[lab1])
        j = random.choice(label_to_idx[lab2])
        sim = F.cosine_similarity(
            all_embeddings[i : i + 1], all_embeddings[j : j + 1],
        ).item()
        sims.append(sim)
        targets.append(0)

    sims = np.array(sims)
    targets = np.array(targets)

    auc = roc_auc_score(targets, sims)

    # EER
    fpr, tpr, _ = roc_curve(targets, sims)
    try:
        from scipy.interpolate import interp1d
        from scipy.optimize import brentq

        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    except ImportError:
        # scipy not available — approximate EER from the ROC curve
        fnr = 1.0 - tpr
        idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float(fpr[idx] + fnr[idx]) / 2.0

    return auc, eer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train pet face verification model")
    p.add_argument("--data", default="dataset/original/dog", help="Root data directory")
    p.add_argument("--output", default="outputs/train", help="Output directory")
    p.add_argument("--network", default="r50", help="Backbone (r50, r101, ir18, ir50, swinb, vitb, ...)")
    p.add_argument("--loss", default="arcface", choices=["arcface", "triplet", "contrastive"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "sgd"])
    p.add_argument("--embedding-size", type=int, default=512)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true", help="Mixed precision training")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=112, help="Input image size (112 for iresnet, 224 for others)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Discover identities ---
    all_ids = sorted(
        d for d in os.listdir(args.data)
        if os.path.isdir(os.path.join(args.data, d))
    )
    print(f"Found {len(all_ids)} identities")

    random.shuffle(all_ids)
    n_val = max(1, int(len(all_ids) * args.val_split))
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]
    print(f"Train: {len(train_ids)} identities | Val: {len(val_ids)} identities")

    # --- Transforms ---
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # --- Datasets ---
    val_dataset = DogDataset(args.data, val_ids, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Model ---
    if args.network.startswith("ir"):
        backbone = get_model(
            args.network, dropout=0.0, fp16=False, num_features=args.embedding_size,
        )
    else:
        backbone = get_model(args.network)
    backbone = backbone.to(device)

    # --- Loss ---
    if args.loss == "arcface":
        criterion = ArcFaceLoss(
            args.embedding_size, train_dataset.num_classes, margin=args.margin,
        ).to(device)
    elif args.loss == "triplet":
        criterion = TripletLoss(margin=args.margin).to(device)
    elif args.loss == "contrastive":
        criterion = ContrastiveLoss(margin=args.margin).to(device)

    # --- Optimizer ---
    params = list(backbone.parameters()) + list(criterion.parameters())
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    os.makedirs(args.output, exist_ok=True)

    # --- Training ---
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        criterion.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.fp16):
                embeddings = backbone(images)
                loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        scheduler.step()
        avg_loss = running_loss / max(len(train_loader), 1)

        # Validation
        auc, eer = validate(backbone, val_dataset, device)
        if auc is not None:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}  AUC={auc:.4f}  EER={eer:.4f}")
        else:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}  (validation skipped)")

        # Checkpoint (compatible with verification.py)
        ckpt = {"state_dict_backbone": backbone.state_dict(), "epoch": epoch}
        torch.save(ckpt, os.path.join(args.output, "last.pt"))

        if auc is not None and auc > best_auc:
            best_auc = auc
            torch.save(ckpt, os.path.join(args.output, "best.pt"))
            print(f"  -> New best AUC: {best_auc:.4f}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Checkpoints saved to {args.output}/")


if __name__ == "__main__":
    main()
