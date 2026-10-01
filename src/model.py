import torch.nn as nn
import torchvision.models as models


def build_model(num_classes=6, pretrained=True):
    """MobileNetV3-Small: ~2.5M params, runs fine on CPU, fine-tunes fast.
    Swap for resnet18/efficientnet_b0 if you have a GPU and want a bit more
    accuracy at the cost of speed."""
    weights = "IMAGENET1K_V1" if pretrained else None
    net = models.mobilenet_v3_small(weights=weights)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net
