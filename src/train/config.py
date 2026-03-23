"""Training configuration for pet face verification."""

import os
from dataclasses import dataclass, field
from typing import Optional

# Project root (two levels up from this file: src/train/config.py -> project root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _abs(path):
    """Make path absolute relative to project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


@dataclass
class TrainConfig:
    # Data (relative to project root)
    train_csv: str = "data/PetFace/split/cats_and_dogs/train.csv"
    val_csv: str = "data/PetFace/split/cats_and_dogs/val.csv"
    verification_csv: str = "data/PetFace/split/cats_and_dogs/verification.csv"
    basedir: str = "data/PetFace/images"
    img_size: int = 224

    # Model
    backbone: str = "convnext_base"  # timm model name
    embedding_size: int = 512
    pretrained: bool = True

    # ArcFace
    arcface_scale: str = "auto"  # "auto" = sqrt(2)*log(num_classes-1), or a float
    arcface_margin: float = 0.5

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps = 256
    num_epochs: int = 20
    num_workers: int = 4
    fp16: bool = True

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    backbone_lr_scale: float = 0.1  # backbone LR = lr * backbone_lr_scale
    weight_decay: float = 0.05

    # Scheduler
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    # Output
    output_dir: str = "outputs/experiment"
    save_every: int = 5  # save checkpoint every N epochs
    seed: int = 42

    # Augmentation
    color_jitter: float = 0.3
    random_erasing: float = 0.1

    # Evaluation
    eval_every: int = 1  # evaluate every N epochs
    label_smoothing: float = 0.0

    # Advanced
    num_subcenters: int = 1  # sub-center ArcFace (K > 1 for noisy/sparse data)
    drop_embed: float = 0.0  # embedding dropout
    strong_aug: bool = False  # enable stronger augmentation

    def __post_init__(self):
        """Resolve relative paths to absolute."""
        self.train_csv = _abs(self.train_csv)
        self.val_csv = _abs(self.val_csv)
        self.verification_csv = _abs(self.verification_csv)
        self.basedir = _abs(self.basedir)
        self.output_dir = _abs(self.output_dir)
