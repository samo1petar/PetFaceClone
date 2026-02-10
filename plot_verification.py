import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score

# Load results

results_dir = 'outputs/christy_dogs_dog2_original_model_2'

df = pd.read_csv(os.path.join(results_dir, 'verification_results.csv'))

# Separate by label
positive = df[df['label'] == 1]['sim']
negative = df[df['label'] == 0]['sim']

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Histogram of similarity scores
ax1 = axes[0]
ax1.hist(negative, bins=50, alpha=0.7, label=f'Different (n={len(negative)})', color='red')
ax1.hist(positive, bins=50, alpha=0.7, label=f'Same (n={len(positive)})', color='green')
ax1.set_xlabel('Cosine Similarity')
ax1.set_ylabel('Count')
ax1.set_title('Similarity Score Distribution')
ax1.legend()
ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='threshold=0.5')

# 2. ROC Curve
ax2 = axes[1]
fpr, tpr, thresholds = roc_curve(df['label'], df['sim'])
roc_auc = auc(fpr, tpr)
ax2.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curve')
ax2.legend(loc='lower right')

# 3. Box plot
ax3 = axes[2]
ax3.boxplot([negative, positive], labels=['Different', 'Same'])
ax3.set_ylabel('Cosine Similarity')
ax3.set_title('Similarity by Category')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'verification_plot.png'), dpi=150)
plt.show()

# Print statistics
print("=" * 50)
print("VERIFICATION RESULTS SUMMARY")
print("=" * 50)
print(f"\nTotal pairs: {len(df)}")
print(f"  - Same identity (label=1): {len(positive)}")
print(f"  - Different identity (label=0): {len(negative)}")

print(f"\nSimilarity Statistics:")
print(f"  Same identity:      mean={positive.mean():.4f}, std={positive.std():.4f}")
print(f"  Different identity: mean={negative.mean():.4f}, std={negative.std():.4f}")

print(f"\nAUC Score: {roc_auc:.4f}")

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")

# Accuracy at optimal threshold
predictions = (df['sim'] >= optimal_threshold).astype(int)
acc = accuracy_score(df['label'], predictions)
print(f"Accuracy at optimal threshold: {acc:.4f}")

# Accuracy at threshold 0.5
predictions_05 = (df['sim'] >= 0.5).astype(int)
acc_05 = accuracy_score(df['label'], predictions_05)
print(f"Accuracy at threshold 0.5: {acc_05:.4f}")

print(f"\nPlot saved to: {results_dir}/verification_plot.png")
