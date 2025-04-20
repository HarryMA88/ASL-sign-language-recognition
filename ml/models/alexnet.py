import torch.nn as nn
from torchvision.models import alexnet
from ml.config import TRAIN_CONFIG

def get_model(num_classes: int=TRAIN_CONFIG["num_classes"]):
    """
    Creates a modified AlexNet model for 28×28 grayscale images.

    Args:
        num_classes (int): Number of output classes for the model.
    """
    model = alexnet(weights=None)

    # 1. Modify first conv layer for grayscale 28×28
    model.features[0] = nn.Conv2d(
        in_channels=1, out_channels=64,
        kernel_size=3, stride=1, padding=1
    )

    # 2. Replace avgpool to handle smaller spatial size (from 6×6 → 1×1)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # 3. Modify classifier for smaller flattened size and fewer params
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Linear(128, num_classes)
    )

    return model