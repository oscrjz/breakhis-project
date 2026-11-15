"""Model builders. Every model ends in a 2-class head: 0 = benign, 1 = malignant."""
import torch.nn as nn
from torchvision import models

SUPPORTED = ("mobilenet_v2", "resnet50", "efficientnet_b0")


def build_model(name, pretrained=True):
    if name == "mobilenet_v2":
        m = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "resnet50":
        m = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    else:
        raise ValueError(f"Unknown model {name}. Choose from {SUPPORTED}")
    return m


def count_params(model):
    return sum(p.numel() for p in model.parameters())
