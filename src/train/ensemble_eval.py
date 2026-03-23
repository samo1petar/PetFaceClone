"""Evaluate an ensemble of models on verification task.

Combines embeddings from multiple models for stronger verification.
Usage:
    uv run python ensemble_eval.py \
        --checkpoints model1.pt:convnext_small model2.pt:eva02_base_patch14_224.mim_in22k
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import VerificationDataset, get_val_transform
from evaluate import embed_with_tta
from model import EmbeddingBackbone


def load_model(checkpoint_path, backbone_name, embedding_size, device):
    """Load a model from checkpoint."""
    backbone = EmbeddingBackbone(backbone_name, embedding_size, pretrained=False).to(device)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["state_dict_backbone"])
    backbone.eval()
    info = f"epoch={ckpt.get('epoch', '?')}, auc={ckpt.get('auc', '?')}"
    print(f"  Loaded {backbone_name} from {checkpoint_path} ({info})")
    return backbone


def compute_embeddings(model, loader, device, use_tta=True):
    """Compute all embeddings for verification pairs."""
    all_emb1 = []
    all_emb2 = []
    all_labels = []

    embed_fn = embed_with_tta if use_tta else lambda m, x: m(x)

    with torch.no_grad():
        for img1, img2, labels in tqdm(loader, desc="Embedding"):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            emb1 = embed_fn(model, img1)
            emb2 = embed_fn(model, img2)
            all_emb1.append(emb1.cpu())
            all_emb2.append(emb2.cpu())
            all_labels.append(labels)

    return torch.cat(all_emb1), torch.cat(all_emb2), torch.cat(all_labels).numpy()


def main():
    parser = argparse.ArgumentParser(description="Ensemble evaluation")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="checkpoint:backbone pairs, e.g. model.pt:convnext_small")
    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--strategy", type=str, default="sim_avg",
                        choices=["sim_avg", "embed_concat", "embed_avg"],
                        help="Ensemble strategy")
    args = parser.parse_args()

    cfg = TrainConfig()
    device = torch.device("cuda")

    # Load verification data
    val_transform = get_val_transform(cfg.img_size)
    verification_dataset = VerificationDataset(cfg.verification_csv, cfg.basedir, val_transform)
    verification_loader = DataLoader(
        verification_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Parse and load models
    models_info = []
    for spec in args.checkpoints:
        parts = spec.split(":")
        ckpt_path = parts[0]
        backbone = parts[1] if len(parts) > 1 else "convnext_small"
        models_info.append((ckpt_path, backbone))

    print(f"Loading {len(models_info)} models...")
    all_embeddings = []

    for ckpt_path, backbone_name in models_info:
        model = load_model(ckpt_path, backbone_name, args.embedding_size, device)
        emb1, emb2, labels = compute_embeddings(
            model, verification_loader, device, use_tta=not args.no_tta,
        )
        all_embeddings.append((emb1, emb2))
        del model
        torch.cuda.empty_cache()

    # Ensemble strategies
    if args.strategy == "sim_avg":
        # Average cosine similarities from each model
        all_sims = []
        for emb1, emb2 in all_embeddings:
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            sim = (emb1 * emb2).sum(dim=1).numpy()
            all_sims.append(sim)
        sims = np.mean(all_sims, axis=0)

    elif args.strategy == "embed_concat":
        # Concatenate embeddings then compute similarity
        emb1_cat = torch.cat([emb1 for emb1, _ in all_embeddings], dim=1)
        emb2_cat = torch.cat([emb2 for _, emb2 in all_embeddings], dim=1)
        emb1_cat = F.normalize(emb1_cat, p=2, dim=1)
        emb2_cat = F.normalize(emb2_cat, p=2, dim=1)
        sims = (emb1_cat * emb2_cat).sum(dim=1).numpy()

    elif args.strategy == "embed_avg":
        # Average embeddings then compute similarity
        emb1_avg = torch.mean(torch.stack([emb1 for emb1, _ in all_embeddings]), dim=0)
        emb2_avg = torch.mean(torch.stack([emb2 for _, emb2 in all_embeddings]), dim=0)
        emb1_avg = F.normalize(emb1_avg, p=2, dim=1)
        emb2_avg = F.normalize(emb2_avg, p=2, dim=1)
        sims = (emb1_avg * emb2_avg).sum(dim=1).numpy()

    # Compute metrics
    auc = roc_auc_score(labels, sims)
    fpr, tpr, thresholds = roc_curve(labels, sims)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds = (sims >= optimal_threshold).astype(int)
    acc = accuracy_score(labels, preds)

    pos_mask = labels == 1
    neg_mask = labels == 0

    print(f"\n{'='*50}")
    print(f"ENSEMBLE RESULTS ({args.strategy}, {len(models_info)} models)")
    print(f"{'='*50}")
    print(f"  AUC:               {auc:.4f} ({auc*100:.2f}%)")
    print(f"  Optimal threshold: {optimal_threshold:.4f}")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  Pos sim:           {sims[pos_mask].mean():.4f} +/- {sims[pos_mask].std():.4f}")
    print(f"  Neg sim:           {sims[neg_mask].mean():.4f} +/- {sims[neg_mask].std():.4f}")
    print(f"{'='*50}")

    # Individual model AUCs for comparison
    if args.strategy == "sim_avg":
        print("\nIndividual model AUCs:")
        for i, (ckpt_path, backbone_name) in enumerate(models_info):
            individual_auc = roc_auc_score(labels, all_sims[i])
            print(f"  {backbone_name}: {individual_auc:.4f} ({individual_auc*100:.2f}%)")


if __name__ == "__main__":
    main()
