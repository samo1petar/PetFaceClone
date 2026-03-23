import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_curve, auc, accuracy_score

from backbones import get_model
from backbones.resnet import r50
from dataset import Verification


def run_verification(args, device):
    """Run pairwise verification and return results DataFrame."""
    if args.timm:
        from train.model import EmbeddingBackbone
        model = EmbeddingBackbone(args.network, embedding_size=args.embedding_size, pretrained=False)
    elif args.network is not None:
        model = get_model(args.network, dropout=0.0, fp16=False, num_features=512)
    else:
        model = r50()
    checkpoint = torch.load(args.weight, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict_backbone" in checkpoint:
        model.load_state_dict(checkpoint["state_dict_backbone"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    dataset = Verification(args.input_csv, args.basedir, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    sim_list = []
    label_list = []
    for img1, img2, labels in tqdm(loader, desc="Verification"):
        with torch.no_grad():
            img1 = img1.to(device)
            img2 = img2.to(device)
            vec1 = F.normalize(model(img1))
            vec2 = F.normalize(model(img2))
            sim = nn.CosineSimilarity()(vec1, vec2).cpu().numpy().tolist()
            sim_list += sim
            label_list += labels.cpu().numpy().tolist()

    df = pd.DataFrame(
        {
            "filename1": dataset.img1_list,
            "filename2": dataset.img2_list,
            "sim": sim_list,
            "label": label_list,
        }
    )
    return df


def compute_metrics(df):
    """Compute AUC, optimal threshold, and accuracies."""
    fpr, tpr, thresholds = roc_curve(df["label"], df["sim"])
    roc_auc = auc(fpr, tpr)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    predictions_opt = (df["sim"] >= optimal_threshold).astype(int)
    acc_opt = accuracy_score(df["label"], predictions_opt)

    predictions_05 = (df["sim"] >= 0.5).astype(int)
    acc_05 = accuracy_score(df["label"], predictions_05)

    positive = df[df["label"] == 1]["sim"]
    negative = df[df["label"] == 0]["sim"]

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc,
        "optimal_threshold": optimal_threshold,
        "acc_optimal": acc_opt,
        "acc_05": acc_05,
        "positive": positive,
        "negative": negative,
    }


def print_summary(df, metrics):
    """Print verification results summary."""
    positive = metrics["positive"]
    negative = metrics["negative"]

    print("=" * 50)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"\nTotal pairs: {len(df)}")
    print(f"  - Same identity (label=1): {len(positive)}")
    print(f"  - Different identity (label=0): {len(negative)}")
    print(f"\nSimilarity Statistics:")
    print(f"  Same identity:      mean={positive.mean():.4f}, std={positive.std():.4f}")
    print(f"  Different identity: mean={negative.mean():.4f}, std={negative.std():.4f}")
    print(f"\nAUC Score: {metrics['auc']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"Accuracy at optimal threshold: {metrics['acc_optimal']:.4f}")
    print(f"Accuracy at threshold 0.5: {metrics['acc_05']:.4f}")


def save_plots(df, metrics, output_dir, plot_suffix=None):
    """Save histogram, ROC curve, and box plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = metrics["positive"]
    negative = metrics["negative"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram
    ax1 = axes[0]
    ax1.hist(negative, bins=50, alpha=0.7, label=f"Different (n={len(negative)})", color="red")
    ax1.hist(positive, bins=50, alpha=0.7, label=f"Same (n={len(positive)})", color="green")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Count")
    ax1.set_title("Similarity Score Distribution")
    ax1.legend()
    ax1.axvline(x=0.5, color="black", linestyle="--", alpha=0.5)

    # ROC Curve
    ax2 = axes[1]
    ax2.plot(metrics["fpr"], metrics["tpr"], color="blue", lw=2, label=f"ROC curve (AUC = {metrics['auc']:.4f})")
    ax2.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend(loc="lower right")

    # Box plot
    ax3 = axes[2]
    ax3.boxplot([negative, positive], labels=["Different", "Same"])
    ax3.set_ylabel("Cosine Similarity")
    ax3.set_title("Similarity by Category")

    plt.tight_layout()
    plot_name = f"verification_plot_{plot_suffix}.png" if plot_suffix else "verification_plot.png"
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


def create_pair_image(img1_path, img2_path):
    """Combine two images side by side."""
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    height = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * height / img1.height), height))
    img2 = img2.resize((int(img2.width * height / img2.height), height))

    combined = Image.new("RGB", (img1.width + img2.width + 10, height), color="white")
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width + 10, 0))
    return combined


def save_wrong_predictions(df, threshold, output_dir, max_images=100):
    """Save side-by-side images of misclassified pairs."""
    df = df.copy()
    df["predicted"] = (df["sim"] >= threshold).astype(int)

    false_positives = df[(df["label"] == 0) & (df["predicted"] == 1)].sort_values("sim", ascending=False)
    false_negatives = df[(df["label"] == 1) & (df["predicted"] == 0)].sort_values("sim", ascending=True)

    fp_dir = os.path.join(output_dir, "false_positives")
    fn_dir = os.path.join(output_dir, "false_negatives")
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    print(f"\nWrong predictions at threshold {threshold}:")
    print(f"  False positives (different predicted as same): {len(false_positives)}")
    print(f"  False negatives (same predicted as different): {len(false_negatives)}")

    for i, (_, row) in enumerate(false_positives.head(max_images).iterrows()):
        id1 = os.path.basename(os.path.dirname(row["filename1"]))
        id2 = os.path.basename(os.path.dirname(row["filename2"]))
        combined = create_pair_image(row["filename1"], row["filename2"])
        combined.save(os.path.join(fp_dir, f"{i:04d}_sim{row['sim']:.3f}_id{id1}_vs_id{id2}.jpg"))

    for i, (_, row) in enumerate(false_negatives.head(max_images).iterrows()):
        id1 = os.path.basename(os.path.dirname(row["filename1"]))
        combined = create_pair_image(row["filename1"], row["filename2"])
        combined.save(os.path.join(fn_dir, f"{i:04d}_sim{row['sim']:.3f}_id{id1}.jpg"))

    accuracy = (df["label"] == df["predicted"]).mean()
    print(f"  Accuracy at threshold {threshold}: {accuracy:.4f}")
    print(f"  Saved to: {output_dir}")


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)

    # 1. Run verification
    df = run_verification(args, device)
    df.to_csv(args.output, index=False)
    print(f"Results saved to: {args.output}")

    # 2. Compute metrics and print summary
    metrics = compute_metrics(df)
    print_summary(df, metrics)

    # 3. Save plots
    plot_suffix = os.path.splitext(os.path.basename(args.output))[0]
    save_plots(df, metrics, output_dir, plot_suffix=plot_suffix)

    # 4. Optionally save wrong predictions
    if args.save_wrong:
        threshold = args.threshold if args.threshold is not None else metrics["optimal_threshold"]
        wrong_dir = os.path.join(output_dir, "wrong_predictions")
        save_wrong_predictions(df, threshold, wrong_dir, args.max_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run verification, compute metrics, plot results")
    parser.add_argument("-w", dest="weight", type=str, required=True, help="Model weights path")
    parser.add_argument("-i", dest="input_csv", type=str, required=True, help="Input pairs CSV")
    parser.add_argument("-o", dest="output", type=str, required=True, help="Output results CSV path")
    parser.add_argument("-b", dest="basedir", type=str, default="data/PetFace/images", help="Base image directory")
    parser.add_argument("--network", type=str, default=None, help="Backbone network name")
    parser.add_argument("--img-size", type=int, default=112, help="Input image size (must match training)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for verification")
    parser.add_argument("--timm", action="store_true", help="Use timm EmbeddingBackbone instead of built-in backbones")
    parser.add_argument("--embedding-size", type=int, default=512, help="Embedding size for timm models")
    parser.add_argument("--save-wrong", action="store_true", help="Save misclassified pair images")
    parser.add_argument("--threshold", type=float, default=None, help="Classification threshold (default: optimal)")
    parser.add_argument("--max-images", type=int, default=100, help="Max wrong prediction images per category")
    main(parser.parse_args())
