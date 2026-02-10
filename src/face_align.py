import numpy as np
import cv2
import skimage
import skimage.transform
import argparse
import os
from pathlib import Path


def align(img,lmk,src,size):
    simtf=skimage.transform.SimilarityTransform()
    simtf.estimate(lmk,src)
    M=simtf.params.copy()
    img_aligned = cv2.warpPerspective(img,M,(size,size),flags=cv2.INTER_AREA)
    return img_aligned

def process_single(args):
    img = cv2.imread(args.img)
    h,w=img.shape[:2]

    src = np.load(args.src)[:, 5:].reshape((5,2))
    tgt = np.load(args.tgt)[0, 5:].reshape((5,2))
    img_aligned = align(img,tgt,src,224)
    cv2.imwrite(args.out,img_aligned)

def process_batch(args):
    src = np.load(args.src).reshape((5,2))

    print(src)

    img_dir = Path(args.img_dir)
    kpts_dir = Path(args.kpts_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for subdir in img_dir.iterdir():
        if not subdir.is_dir():
            continue

        out_subdir = out_dir / subdir.name
        out_subdir.mkdir(parents=True, exist_ok=True)

        for img_path in subdir.glob("*.jpg"):
            kpts_path = kpts_dir / subdir.name / (img_path.stem + ".npy")
            out_path = out_subdir / img_path.name

            if not kpts_path.exists():
                print(f"Skipping {img_path}: keypoints not found at {kpts_path}")
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipping {img_path}: could not read image")
                continue

            tgt = np.load(str(kpts_path))[0, 5:].reshape((5,2)) * 224

            img_aligned = align(img, tgt, src, 224)
            cv2.imwrite(str(out_path), img_aligned)
            print(f"Aligned: {img_path.name} -> {out_path}")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source landmarks path", required=True)
    parser.add_argument("--tgt", type=str, help="your image's landmarks path (single image mode)")
    parser.add_argument("--img", type=str, help="your image path (single image mode)")
    parser.add_argument("--out", type=str, help="output path (single image mode)")
    parser.add_argument("--img_dir", type=str, help="input images directory (batch mode)")
    parser.add_argument("--kpts_dir", type=str, help="keypoints directory (batch mode)")
    parser.add_argument("--out_dir", type=str, help="output directory (batch mode)")

    args = parser.parse_args()

    if args.img_dir and args.kpts_dir and args.out_dir:
        process_batch(args)
    elif args.tgt and args.img and args.out:
        process_single(args)
    else:
        parser.error("Provide either (--img_dir, --kpts_dir, --out_dir) for batch mode or (--tgt, --img, --out) for single image mode")
