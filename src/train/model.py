"""Model definition: pretrained backbone + embedding head + ArcFace classifier."""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingBackbone(nn.Module):
    """Pretrained backbone from timm with an embedding projection head."""

    def __init__(self, backbone_name="convnext_base", embedding_size=512, pretrained=True,
                 drop_embed=0.0):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0,  # remove classifier
        )
        # Detect actual output dim (num_features can be wrong for some models)
        with torch.no_grad():
            dummy = torch.randn(2, 3, 224, 224)
            feature_dim = self.backbone(dummy).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feature_dim, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self.drop = nn.Dropout(drop_embed) if drop_embed > 0 else nn.Identity()

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.head(features)
        embedding = self.drop(embedding)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class ArcFaceClassifier(nn.Module):
    """ArcFace margin-based classifier for metric learning.

    Given L2-normalized embeddings and labels, applies angular margin penalty
    and returns scaled logits for cross-entropy loss.

    Supports sub-center ArcFace (K > 1) for handling noisy/sparse labels.
    """

    def __init__(self, embedding_size, num_classes, scale=None, margin=0.5,
                 label_smoothing=0.0, num_subcenters=1):
        super().__init__()
        # Auto-compute scale if not provided: s = sqrt(2) * log(C-1)
        if scale is None:
            self.scale = math.sqrt(2) * math.log(max(num_classes - 1, 1))
        else:
            self.scale = scale

        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability
        self.threshold = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.num_subcenters = num_subcenters

        if num_subcenters > 1:
            # Sub-center ArcFace: K sub-centers per class
            self.weight = nn.Parameter(torch.empty(num_classes * num_subcenters, embedding_size))
        else:
            self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, embeddings, labels):
        norm_weight = F.normalize(self.weight, p=2, dim=1)

        if self.num_subcenters > 1:
            # Sub-center: compute cos for all sub-centers, take max per class
            cosine_all = F.linear(embeddings, norm_weight)  # (B, C*K)
            cosine_all = cosine_all.clamp(-1, 1)
            # Reshape to (B, C, K) and take max over sub-centers
            cosine_all = cosine_all.view(-1, self.num_classes, self.num_subcenters)
            cosine, _ = cosine_all.max(dim=2)  # (B, C)
        else:
            cosine = F.linear(embeddings, norm_weight)
            cosine = cosine.clamp(-1, 1)

        # Apply ArcFace margin to target class
        sin_theta = torch.sqrt(1.0 - cosine.pow(2))
        cos_theta_m = cosine * self.cos_m - sin_theta * self.sin_m
        # Safe fallback when cos(theta) < threshold
        cos_theta_m = torch.where(cosine > self.threshold, cos_theta_m, cosine - self.mm)

        # One-hot target mask
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        logits = torch.where(one_hot.bool(), cos_theta_m, cosine)
        logits = logits * self.scale

        loss = self.ce_loss(logits, labels)
        return loss
