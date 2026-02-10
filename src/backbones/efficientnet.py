import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_v2_m, EfficientNet_V2_M_Weights,
    efficientnet_v2_l, EfficientNet_V2_L_Weights
)

def efficientnetb0():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    head_bn = nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    # EfficientNet B0 has 1280 features before the classifier
    model.classifier = nn.Sequential(nn.Linear(1280, 512), head_bn)

    return model

def efficientnetv2s():
    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    head_bn = nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    # EfficientNet V2-S has 1280 features before the classifier
    model.classifier = nn.Sequential(nn.Linear(1280, 512), head_bn)

    return model

def efficientnetv2m():
    weights = EfficientNet_V2_M_Weights.DEFAULT
    model = efficientnet_v2_m(weights=weights)
    head_bn = nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    # EfficientNet V2-M has 1280 features before the classifier
    model.classifier = nn.Sequential(nn.Linear(1280, 512), head_bn)

    return model

def efficientnetv2l():
    weights = EfficientNet_V2_L_Weights.DEFAULT
    model = efficientnet_v2_l(weights=weights)
    head_bn = nn.BatchNorm1d(512, eps=1e-05)
    nn.init.constant_(head_bn.weight, 1.0)
    head_bn.weight.requires_grad = False
    # EfficientNet V2-L has 1280 features before the classifier
    model.classifier = nn.Sequential(nn.Linear(1280, 512), head_bn)

    return model
