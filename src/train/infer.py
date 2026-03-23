"""Inference script for pet face verification using ONNX Runtime.

Given two images, compute cosine similarity to determine if they are the same animal.

Usage:
    uv run python infer.py \
        --model ../../outputs/exp10_effnetv2b0/model.onnx \
        --img1 /path/to/image1.jpg \
        --img2 /path/to/image2.jpg

    # With threshold (default 0.5):
    uv run python infer.py \
        --model model.onnx \
        --img1 img1.jpg --img2 img2.jpg \
        --threshold 0.45
"""

import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image_path, img_size=224):
    """Load and preprocess an image for the model."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img[np.newaxis, ...]  # add batch dim


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description="Pet face verification inference")
    parser.add_argument("--model", type=str, required=True, help="ONNX model path")
    parser.add_argument("--img1", type=str, required=True)
    parser.add_argument("--img2", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Similarity threshold for same-animal decision")
    parser.add_argument("--tta", action="store_true", help="Use test-time augmentation (flip)")
    args = parser.parse_args()

    # Load model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(args.model, providers=providers)
    active_provider = session.get_providers()[0]
    print(f"Model: {args.model}")
    print(f"Provider: {active_provider}")

    # Preprocess images
    img1 = preprocess(args.img1, args.img_size)
    img2 = preprocess(args.img2, args.img_size)

    # Run inference
    start = time.perf_counter()
    emb1 = session.run(None, {"input": img1})[0][0]
    emb2 = session.run(None, {"input": img2})[0][0]

    if args.tta:
        # Average with horizontally flipped version
        img1_flip = img1[:, :, :, ::-1].copy()
        img2_flip = img2[:, :, :, ::-1].copy()
        emb1_flip = session.run(None, {"input": img1_flip})[0][0]
        emb2_flip = session.run(None, {"input": img2_flip})[0][0]
        emb1 = emb1 + emb1_flip
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb2 + emb2_flip
        emb2 = emb2 / np.linalg.norm(emb2)

    elapsed = (time.perf_counter() - start) * 1000

    sim = cosine_similarity(emb1, emb2)
    same = sim >= args.threshold

    print(f"\nImage 1: {args.img1}")
    print(f"Image 2: {args.img2}")
    print(f"Cosine similarity: {sim:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Result: {'SAME animal' if same else 'DIFFERENT animals'}")
    print(f"Inference time: {elapsed:.1f} ms")


if __name__ == "__main__":
    main()
