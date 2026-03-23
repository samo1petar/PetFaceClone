"""Dataset classes for pet face verification training and evaluation."""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_train_transform(img_size=224, color_jitter=0.3, random_erasing=0.1, strong=False):
    """Training augmentation pipeline."""
    aug_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(
            brightness=color_jitter, contrast=color_jitter,
            saturation=color_jitter, hue=color_jitter * 0.3,
        ),
        transforms.RandomGrayscale(p=0.05),
    ]
    if strong:
        aug_list.extend([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
        ])
    aug_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=random_erasing),
    ])
    return transforms.Compose(aug_list)


def get_val_transform(img_size=224):
    """Validation transform (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ClassificationDataset(Dataset):
    """Dataset for ArcFace training: returns (image, label)."""

    def __init__(self, csv_path, basedir, transform=None):
        df = pd.read_csv(csv_path)
        self.image_paths = [os.path.join(basedir, p) for p in df["filename"]]
        self.labels = df["label"].tolist()
        self.num_classes = df["label"].nunique()
        self.transform = transform
        print(f"Loaded {len(self)} images, {self.num_classes} classes from {csv_path}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


class VerificationDataset(Dataset):
    """Dataset for verification evaluation: returns (img1, img2, label)."""

    def __init__(self, csv_path, basedir, transform=None):
        df = pd.read_csv(csv_path)
        self.img1_paths = [os.path.join(basedir, p) for p in df["filename1"]]
        self.img2_paths = [os.path.join(basedir, p) for p in df["filename2"]]
        self.labels = df["label"].tolist()
        # Extract species from filenames (e.g., "cat/12345/00.png" -> "cat")
        self.species = [f.split("/")[0] for f in df["filename1"]]
        self.transform = transform or get_val_transform()
        print(f"Loaded {len(self)} verification pairs from {csv_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img1 = Image.open(self.img1_paths[idx]).convert("RGB")
        img2 = Image.open(self.img2_paths[idx]).convert("RGB")
        label = self.labels[idx]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.long)
