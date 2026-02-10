import pandas as pd
import numpy as np
from PIL import Image
import os
import argparse

def create_pair_image(img1_path, img2_path):
    """Combine two images side by side."""
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # Resize to same height
    height = max(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * height / img1.height), height))
    img2 = img2.resize((int(img2.width * height / img2.height), height))

    # Combine side by side
    combined = Image.new('RGB', (img1.width + img2.width + 10, height), color='white')
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width + 10, 0))

    return combined

def main(args):
    # Load results
    df = pd.read_csv(args.input)

    # Predictions based on threshold
    df['predicted'] = (df['sim'] >= args.threshold).astype(int)

    # Find wrong predictions
    false_positives = df[(df['label'] == 0) & (df['predicted'] == 1)]  # Different but predicted same
    false_negatives = df[(df['label'] == 1) & (df['predicted'] == 0)]  # Same but predicted different

    print(f"Threshold: {args.threshold}")
    print(f"Total pairs: {len(df)}")
    print(f"False positives (different predicted as same): {len(false_positives)}")
    print(f"False negatives (same predicted as different): {len(false_negatives)}")

    # Create output directories
    fp_dir = os.path.join(args.output, 'false_positives')
    fn_dir = os.path.join(args.output, 'false_negatives')
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    # Sort by similarity to get most egregious errors
    false_positives = false_positives.sort_values('sim', ascending=False)
    false_negatives = false_negatives.sort_values('sim', ascending=True)

    # Limit number of saved images
    max_save = args.max_images

    # Save false positives
    print(f"\nSaving up to {max_save} false positives...")
    for i, (_, row) in enumerate(false_positives.head(max_save).iterrows()):
        img1_path = row['filename1']
        img2_path = row['filename2']
        sim = row['sim']

        # Extract directory names (identity folders)
        id1 = os.path.basename(os.path.dirname(img1_path))
        id2 = os.path.basename(os.path.dirname(img2_path))

        combined = create_pair_image(img1_path, img2_path)
        output_path = os.path.join(fp_dir, f'{i:04d}_sim{sim:.3f}_id{id1}_vs_id{id2}.jpg')
        combined.save(output_path)

    print(f"Saved {min(len(false_positives), max_save)} false positive pairs to {fp_dir}")

    # Save false negatives
    print(f"\nSaving up to {max_save} false negatives...")
    for i, (_, row) in enumerate(false_negatives.head(max_save).iterrows()):
        img1_path = row['filename1']
        img2_path = row['filename2']
        sim = row['sim']

        # Extract directory name (identity folder)
        id1 = os.path.basename(os.path.dirname(img1_path))

        combined = create_pair_image(img1_path, img2_path)
        output_path = os.path.join(fn_dir, f'{i:04d}_sim{sim:.3f}_id{id1}.jpg')
        combined.save(output_path)

    print(f"Saved {min(len(false_negatives), max_save)} false negative pairs to {fn_dir}")

    # Print summary
    accuracy = (df['label'] == df['predicted']).mean()
    print(f"\nOverall accuracy at threshold {args.threshold}: {accuracy:.4f}")
    print(f"\nOutput saved to: {args.output}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save wrongly classified verification pairs')
    parser.add_argument('-i', '--input', type=str, default='outputs/christy_dogs_input_size_224/verification_results.csv',
                        help='Input CSV file with verification results')
    parser.add_argument('-o', '--output', type=str, default='outputs/christy_dogs_input_size_224/wrong_predictions',
                        help='Output directory for wrong predictions')
    parser.add_argument('-t', '--threshold', type=float, default=0.4,
                        help='Similarity threshold for classification')
    parser.add_argument('-m', '--max-images', type=int, default=100,
                        help='Maximum number of images to save per category')

    main(parser.parse_args())
