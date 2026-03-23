"""Analyze verification errors from best model to understand failure modes."""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TrainConfig
from dataset import VerificationDataset, get_val_transform
from evaluate import embed_with_tta
from model import EmbeddingBackbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="convnext_small")
    parser.add_argument("--embedding-size", type=int, default=512)
    args = parser.parse_args()

    cfg = TrainConfig()
    device = torch.device("cuda")

    # Load model
    backbone = EmbeddingBackbone(args.backbone, args.embedding_size, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    backbone.load_state_dict(ckpt["state_dict_backbone"])
    backbone.eval()
    print(f"Loaded: {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, AUC {ckpt.get('auc', '?')})")

    # Load verification data
    val_transform = get_val_transform(cfg.img_size)
    verification_dataset = VerificationDataset(cfg.verification_csv, cfg.basedir, val_transform)
    verification_loader = DataLoader(
        verification_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True,
    )

    # Compute all similarities with TTA
    all_sims = []
    all_labels = []
    with torch.no_grad():
        for img1, img2, labels in tqdm(verification_loader, desc="Computing embeddings"):
            img1 = img1.to(device, non_blocking=True)
            img2 = img2.to(device, non_blocking=True)
            emb1 = embed_with_tta(backbone, img1)
            emb2 = embed_with_tta(backbone, img2)
            sim = (emb1 * emb2).sum(dim=1).cpu().numpy()
            all_sims.append(sim)
            all_labels.append(labels.numpy())

    sims = np.concatenate(all_sims)
    labels = np.concatenate(all_labels)

    # Load CSV for filenames
    df = pd.read_csv(cfg.verification_csv)

    # Find optimal threshold
    from sklearn.metrics import roc_auc_score, roc_curve
    auc = roc_auc_score(labels, sims)
    fpr, tpr, thresholds = roc_curve(labels, sims)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nAUC: {auc:.4f} ({auc*100:.2f}%)")
    print(f"Optimal threshold: {optimal_threshold:.4f}")

    # Error analysis
    preds = (sims >= optimal_threshold).astype(int)
    errors = preds != labels

    # Positive pair errors (same animal, predicted different)
    pos_mask = labels == 1
    false_negatives = errors & pos_mask
    fn_sims = sims[false_negatives]

    # Negative pair errors (different animal, predicted same)
    neg_mask = labels == 0
    false_positives = errors & neg_mask
    fp_sims = sims[false_positives]

    print(f"\nTotal pairs: {len(labels)}")
    print(f"  Positive: {pos_mask.sum()}, Negative: {neg_mask.sum()}")
    print(f"\nErrors at optimal threshold ({optimal_threshold:.4f}):")
    print(f"  False Negatives (missed same-animal): {false_negatives.sum()} "
          f"({false_negatives.sum()/pos_mask.sum()*100:.2f}% of positives)")
    print(f"  False Positives (wrong same-animal): {false_positives.sum()} "
          f"({false_positives.sum()/neg_mask.sum()*100:.2f}% of negatives)")

    # Similarity distributions
    pos_sims = sims[pos_mask]
    neg_sims = sims[neg_mask]
    print(f"\nSimilarity distributions:")
    print(f"  Positive: mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}, "
          f"min={pos_sims.min():.4f}, median={np.median(pos_sims):.4f}")
    print(f"  Negative: mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}, "
          f"max={neg_sims.max():.4f}, median={np.median(neg_sims):.4f}")

    # Gap analysis
    overlap_pos = (pos_sims < optimal_threshold).sum()
    overlap_neg = (neg_sims >= optimal_threshold).sum()
    print(f"\nOverlap region:")
    print(f"  Positives below threshold: {overlap_pos} ({overlap_pos/len(pos_sims)*100:.2f}%)")
    print(f"  Negatives above threshold: {overlap_neg} ({overlap_neg/len(neg_sims)*100:.2f}%)")

    # Hardest false negatives (same animal but low similarity)
    if false_negatives.sum() > 0:
        fn_indices = np.where(false_negatives)[0]
        fn_sort = np.argsort(sims[fn_indices])  # lowest sim first
        print(f"\nHardest false negatives (same animal, low sim):")
        for i in fn_sort[:10]:
            idx = fn_indices[i]
            print(f"  sim={sims[idx]:.4f}: {df.iloc[idx]['filename1']} vs {df.iloc[idx]['filename2']}")

    # Hardest false positives (different animal but high similarity)
    if false_positives.sum() > 0:
        fp_indices = np.where(false_positives)[0]
        fp_sort = np.argsort(-sims[fp_indices])  # highest sim first
        print(f"\nHardest false positives (diff animal, high sim):")
        for i in fp_sort[:10]:
            idx = fp_indices[i]
            print(f"  sim={sims[idx]:.4f}: {df.iloc[idx]['filename1']} vs {df.iloc[idx]['filename2']}")

    # Species breakdown
    def get_species(filename):
        parts = filename.split("/")
        if len(parts) >= 2:
            return parts[0].lower()
        return "unknown"

    df["species1"] = df["filename1"].apply(get_species)
    df["species2"] = df["filename2"].apply(get_species)
    df["same_species"] = df["species1"] == df["species2"]
    df["sim"] = sims
    df["label"] = labels
    df["error"] = errors

    print(f"\nPer-species breakdown:")
    for sp1 in df["species1"].unique():
        mask = (df["species1"] == sp1) & (df["species2"] == sp1)
        if mask.sum() == 0:
            continue
        sub = df[mask]
        sub_auc = roc_auc_score(sub["label"], sub["sim"]) if sub["label"].nunique() > 1 else 0
        error_rate = sub["error"].mean()
        print(f"  {sp1}-{sp1}: {mask.sum()} pairs, AUC={sub_auc:.4f}, "
              f"error_rate={error_rate*100:.2f}%")


if __name__ == "__main__":
    main()
