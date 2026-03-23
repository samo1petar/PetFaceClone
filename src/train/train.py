"""Main training script for pet face verification using ArcFace.

Usage:
    uv run python src/train/train.py [--options]
"""

import argparse
import json
import math
import os
import time

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


def create_optimizer(backbone, classifier, cfg):
    """Create optimizer with lower LR for pretrained backbone."""
    param_groups = [
        {"params": backbone.head.parameters(), "lr": cfg.lr},
        {"params": backbone.backbone.parameters(), "lr": cfg.lr * cfg.backbone_lr_scale},
        {"params": classifier.parameters(), "lr": cfg.lr},
    ]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def create_scheduler(optimizer, cfg, steps_per_epoch):
    """Cosine annealing with warmup."""
    warmup_steps = cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.num_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_plots(train_losses, val_aucs, output_dir):
    """Save training loss and validation AUC plots."""
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


def train(cfg: TrainConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(vars(cfg), f, indent=2)

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(cfg.output_dir, "tensorboard"))

    # Data
    train_transform = get_train_transform(cfg.img_size, cfg.color_jitter, cfg.random_erasing,
                                          strong=cfg.strong_aug)
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
    backbone = EmbeddingBackbone(
        cfg.backbone, cfg.embedding_size, cfg.pretrained, drop_embed=cfg.drop_embed,
    ).to(device)
    num_classes = train_dataset.num_classes

    scale = None
    if cfg.arcface_scale != "auto":
        scale = float(cfg.arcface_scale)
    classifier = ArcFaceClassifier(
        cfg.embedding_size, num_classes, scale=scale, margin=cfg.arcface_margin,
        label_smoothing=cfg.label_smoothing, num_subcenters=cfg.num_subcenters,
    ).to(device)

    print(f"\nBackbone: {cfg.backbone}")
    print(f"  Params: {sum(p.numel() for p in backbone.parameters()) / 1e6:.1f}M")
    print(f"  Embedding: {cfg.embedding_size}, drop={cfg.drop_embed}")
    print(f"ArcFace scale: {classifier.scale:.2f}, margin: {cfg.arcface_margin}, "
          f"subcenters: {cfg.num_subcenters}")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")
    print(f"Training: {len(train_dataset)} images, {num_classes} classes")
    print(f"Verification: {len(verification_dataset)} pairs")
    print(f"Strong aug: {cfg.strong_aug}")
    print()

    # Optimizer and scheduler
    optimizer = create_optimizer(backbone, classifier, cfg)
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    scheduler = create_scheduler(optimizer, cfg, steps_per_epoch)

    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16)

    # Training loop
    train_losses = []
    val_aucs = []
    best_auc = 0.0
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
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item() * cfg.grad_accum_steps:.4f}", lr=f"{current_lr:.6f}")

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)
        writer.add_scalar("train/loss", avg_loss, epoch + 1)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")

        # Verification evaluation
        should_eval = ((epoch + 1) % cfg.eval_every == 0) or (epoch + 1 == cfg.num_epochs)
        if should_eval:
            use_tta = ((epoch + 1) % 5 == 0) or (epoch + 1 == cfg.num_epochs)
            metrics = compute_verification_metrics(backbone, verification_loader, device, use_tta=use_tta)
            tta_label = " [TTA]" if use_tta else ""
            print(f"Evaluation{tta_label}:")
            print_metrics(metrics)
            val_aucs.append((epoch + 1, metrics["auc"]))
            writer.add_scalar("val/auc", metrics["auc"], epoch + 1)
            writer.add_scalar("val/acc_optimal", metrics["acc_optimal"], epoch + 1)
            for key in metrics:
                if key.startswith("auc_"):
                    writer.add_scalar(f"val/{key}", metrics[key], epoch + 1)

            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(
                    {"state_dict_backbone": backbone.state_dict(),
                     "epoch": epoch + 1, "auc": best_auc},
                    os.path.join(cfg.output_dir, "model_best.pt"),
                )
                print(f"  -> New best AUC: {best_auc:.4f} ({best_auc*100:.2f}%)")

        # Always save periodic checkpoint
        if (epoch + 1) % cfg.save_every == 0:
            torch.save(
                {"state_dict_backbone": backbone.state_dict(),
                 "state_dict_classifier": classifier.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1},
                os.path.join(cfg.output_dir, f"checkpoint_epoch{epoch+1}.pt"),
            )

        if val_aucs:
            save_plots(train_losses, val_aucs, cfg.output_dir)

    # Final evaluation with TTA if not already done
    if not val_aucs or val_aucs[-1][0] != cfg.num_epochs:
        metrics = compute_verification_metrics(backbone, verification_loader, device, use_tta=True)
        print("Final evaluation [TTA]:")
        print_metrics(metrics)
        val_aucs.append((cfg.num_epochs, metrics["auc"]))
        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]

    final_auc = val_aucs[-1][1] if val_aucs else 0.0

    # Save final model
    torch.save(
        {"state_dict_backbone": backbone.state_dict(),
         "epoch": cfg.num_epochs, "auc": final_auc},
        os.path.join(cfg.output_dir, "model_last.pt"),
    )

    save_plots(train_losses, val_aucs, cfg.output_dir)

    # Final summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Best AUC: {best_auc:.4f} ({best_auc*100:.2f}%)")
    print(f"Final AUC: {final_auc:.4f} ({final_auc*100:.2f}%)")
    print(f"Models saved in: {cfg.output_dir}")

    writer.close()
    return best_auc


def parse_args():
    parser = argparse.ArgumentParser(description="Pet face verification training")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--backbone-lr-scale", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None, help="Evaluate every N epochs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--num-subcenters", type=int, default=None)
    parser.add_argument("--drop-embed", type=float, default=None)
    parser.add_argument("--strong-aug", action="store_true", default=None)
    parser.add_argument("--scale", type=float, default=None, help="ArcFace scale (overrides auto)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig()

    # Override config with CLI args
    if args.backbone:
        cfg.backbone = args.backbone
    if args.embedding_size:
        cfg.embedding_size = args.embedding_size
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.grad_accum:
        cfg.grad_accum_steps = args.grad_accum
    if args.lr:
        cfg.lr = args.lr
    if args.backbone_lr_scale:
        cfg.backbone_lr_scale = args.backbone_lr_scale
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.warmup_epochs is not None:
        cfg.warmup_epochs = args.warmup_epochs
    if args.margin:
        cfg.arcface_margin = args.margin
    if args.optimizer:
        cfg.optimizer = args.optimizer
    if args.output:
        cfg.output_dir = args.output
    if args.fp16:
        cfg.fp16 = True
    if args.no_fp16:
        cfg.fp16 = False
    if args.seed:
        cfg.seed = args.seed
    if args.eval_every:
        cfg.eval_every = args.eval_every
    if args.label_smoothing is not None:
        cfg.label_smoothing = args.label_smoothing
    if args.num_subcenters is not None:
        cfg.num_subcenters = args.num_subcenters
    if args.drop_embed is not None:
        cfg.drop_embed = args.drop_embed
    if args.strong_aug:
        cfg.strong_aug = True
    if args.scale is not None:
        cfg.arcface_scale = str(args.scale)

    # Set seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True

    train(cfg)


if __name__ == "__main__":
    main()
