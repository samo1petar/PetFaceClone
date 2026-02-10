import torch
from torch import nn
from torchvision.models import (
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights
)

def mobilenetv3s():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    head_bn = nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    # MobileNet V3 Small has 576 features before the classifier
    model.classifier = nn.Sequential(nn.Linear(576, 512), head_bn)

    return model

def mobilenetv3l():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    head_bn = nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    # MobileNet V3 Large has 960 features before the classifier
    model.classifier = nn.Sequential(nn.Linear(960, 512), head_bn)

    return model
