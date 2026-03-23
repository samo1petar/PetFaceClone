"""Evaluation: compute verification AUC and other metrics."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


def embed_with_tta(model, images):
    """Compute embedding as average of original + horizontally flipped image."""
    emb_orig = model(images)
    emb_flip = model(torch.flip(images, dims=[3]))  # flip width dimension
    emb = F.normalize(emb_orig + emb_flip, p=2, dim=1)
    return emb


def compute_verification_metrics(model, verification_loader, device, use_tta=False):
    """Compute verification metrics using cosine similarity.

    Returns dict with AUC, optimal threshold, accuracies, similarity stats,
    and per-species AUC if species info is available.
    """
    model.eval()
    all_sims = []
    all_labels = []

    embed_fn = embed_with_tta if use_tta else lambda m, x: F.normalize(m(x), p=2, dim=1)

    with torch.no_grad():
        for img1, img2, labels in tqdm(verification_loader, desc="Evaluating"):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)

            emb1 = embed_fn(model, img1)
            emb2 = embed_fn(model, img2)

            sim = (emb1 * emb2).sum(dim=1).cpu().numpy()
            all_sims.append(sim)
            all_labels.append(labels.numpy())

    sims = np.concatenate(all_sims)
    labels = np.concatenate(all_labels)

    # AUC
    auc = roc_auc_score(labels, sims)

    # Optimal threshold (Youden's J statistic)
    fpr, tpr, thresholds = roc_curve(labels, sims)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Accuracy at optimal threshold
    preds_opt = (sims >= optimal_threshold).astype(int)
    acc_opt = accuracy_score(labels, preds_opt)

    # Accuracy at threshold 0.5
    preds_05 = (sims >= 0.5).astype(int)
    acc_05 = accuracy_score(labels, preds_05)

    # Similarity statistics
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_sims = sims[pos_mask]
    neg_sims = sims[neg_mask]

    metrics = {
        "auc": auc,
        "optimal_threshold": optimal_threshold,
        "acc_optimal": acc_opt,
        "acc_05": acc_05,
        "pos_sim_mean": pos_sims.mean(),
        "pos_sim_std": pos_sims.std(),
        "neg_sim_mean": neg_sims.mean(),
        "neg_sim_std": neg_sims.std(),
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "sims": sims,
        "labels": labels,
    }

    # Per-species AUC
    dataset = verification_loader.dataset
    if hasattr(dataset, "species"):
        species = np.array(dataset.species)
        for sp in sorted(set(species)):
            mask = species == sp
            if mask.sum() > 0 and len(set(labels[mask])) > 1:
                sp_auc = roc_auc_score(labels[mask], sims[mask])
                metrics[f"auc_{sp}"] = sp_auc

    model.train()
    return metrics


def print_metrics(metrics):
    """Pretty-print verification metrics."""
    print("=" * 50)
    print("VERIFICATION METRICS")
    print("=" * 50)
    print(f"  AUC:                {metrics['auc']:.4f} ({metrics['auc']*100:.2f}%)")
    # Per-species AUC
    for key in sorted(metrics):
        if key.startswith("auc_"):
            sp = key[4:]
            print(f"  AUC ({sp}):         {metrics[key]:.4f} ({metrics[key]*100:.2f}%)")
    print(f"  Optimal threshold:  {metrics['optimal_threshold']:.4f}")
    print(f"  Acc @ optimal:      {metrics['acc_optimal']:.4f}")
    print(f"  Pos sim:            {metrics['pos_sim_mean']:.4f} +/- {metrics['pos_sim_std']:.4f}")
    print(f"  Neg sim:            {metrics['neg_sim_mean']:.4f} +/- {metrics['neg_sim_std']:.4f}")
    print("=" * 50)
