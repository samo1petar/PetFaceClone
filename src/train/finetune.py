"""Fine-tune a pretrained model with low learning rate.

Loads a checkpoint and continues training with reduced LR for more epochs.
Usage:
    uv run python finetune.py --checkpoint ../../outputs/exp2_convnext_small/model_best.pt \
        --backbone convnext_small --output ../../outputs/exp5_finetune
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import TrainConfig
from dataset import (
    ClassificationDataset,
    VerificationDataset,
    get_train_transform,
    get_val_transform,
)
from evaluate import compute_verification_metrics, print_metrics
from model import ArcFaceClassifier, EmbeddingBackbone


def save_plots(train_losses, val_aucs, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)
    if val_aucs:
        epochs, aucs = zip(*val_aucs)
        ax2.plot(epochs, [a * 100 for a in aucs], marker="o", label="Val AUC (%)")
        ax2.axhline(y=99.5, color="r", linestyle="--", label="Target (99.5%)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC (%)")
        ax2.set_title("Validation AUC")
        ax2.legend()
        ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_plots.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="convnext_small")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--eval-every", type=int, default=2)
    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.backbone = args.backbone
    cfg.embedding_size = args.embedding_size
    cfg.batch_size = args.batch_size
    cfg.grad_accum_steps = args.grad_accum
    cfg.lr = args.lr
    cfg.backbone_lr_scale = args.backbone_lr_scale
    cfg.num_epochs = args.epochs
    cfg.arcface_margin = args.margin
    cfg.output_dir = args.output
    cfg.warmup_epochs = 1
    cfg.eval_every = args.eval_every

    device = torch.device("cuda")
    os.makedirs(cfg.output_dir, exist_ok=True)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, **vars(cfg)}, f, indent=2, default=str)

    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))

    # Data
    train_transform = get_train_transform(cfg.img_size, cfg.color_jitter, cfg.random_erasing)
    val_transform = get_val_transform(cfg.img_size)

    train_dataset = ClassificationDataset(cfg.train_csv, cfg.basedir, train_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )

    verification_dataset = VerificationDataset(cfg.verification_csv, cfg.basedir, val_transform)
    verification_loader = DataLoader(
        verification_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    # Model
    backbone = EmbeddingBackbone(cfg.backbone, cfg.embedding_size, pretrained=False).to(device)
    num_classes = train_dataset.num_classes

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["state_dict_backbone"])
    print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, AUC {ckpt.get('auc', '?')})")

    classifier = ArcFaceClassifier(
        cfg.embedding_size, num_classes, margin=cfg.arcface_margin,
    ).to(device)

    print(f"\nBackbone: {cfg.backbone} (fine-tuning)")
    print(f"  Embedding: {cfg.embedding_size}")
    print(f"  LR: {cfg.lr}, Backbone LR: {cfg.lr * cfg.backbone_lr_scale}")
    print(f"  Margin: {cfg.arcface_margin}")
    print(f"  Effective batch: {cfg.batch_size * cfg.grad_accum_steps}")
    print()

    # Initial evaluation
    print("Pre-finetune evaluation:")
    metrics = compute_verification_metrics(backbone, verification_loader, device, use_tta=True)
    print_metrics(metrics)

    # Optimizer - lower LR, backbone gets even lower
    param_groups = [
        {"params": backbone.head.parameters(), "lr": cfg.lr},
        {"params": backbone.backbone.parameters(), "lr": cfg.lr * cfg.backbone_lr_scale},
        {"params": classifier.parameters(), "lr": cfg.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)

    # Training loop
    train_losses = []
    val_aucs = []
    best_auc = metrics["auc"]
    global_step = 0

    for epoch in range(cfg.num_epochs):
        backbone.train()
        classifier.train()
        epoch_loss = 0.0
        num_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        optimizer.zero_grad()

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=cfg.fp16):
                embeddings = backbone(images)
                loss = classifier(embeddings, labels)
                loss = loss / cfg.grad_accum_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            epoch_loss += loss.item() * cfg.grad_accum_steps
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item() * cfg.grad_accum_steps:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        # Evaluation
        should_eval = ((epoch + 1) % cfg.eval_every == 0) or (epoch + 1 == cfg.num_epochs)
        if should_eval:
            use_tta = ((epoch + 1) % 5 == 0) or (epoch + 1 == cfg.num_epochs)
            metrics = compute_verification_metrics(backbone, verification_loader, device, use_tta=use_tta)
            print_metrics(metrics)
            val_aucs.append((epoch + 1, metrics["auc"]))
            writer.add_scalar("val/auc", metrics["auc"], epoch + 1)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(
                    {"state_dict_backbone": backbone.state_dict(),
                     "epoch": epoch + 1, "auc": best_auc},
                    os.path.join(cfg.output_dir, "model_best.pt"),
                )
                print(f"  -> New best AUC: {best_auc:.4f} ({best_auc*100:.2f}%)")

        if val_aucs:
            save_plots(train_losses, val_aucs, cfg.output_dir)

    # Final
    torch.save(
        {"state_dict_backbone": backbone.state_dict(),
         "epoch": cfg.num_epochs, "auc": val_aucs[-1][1] if val_aucs else 0},
        os.path.join(cfg.output_dir, "model_last.pt"),
    )
    print(f"\nBest AUC: {best_auc:.4f} ({best_auc*100:.2f}%)")
    writer.close()


if __name__ == "__main__":
    main()
