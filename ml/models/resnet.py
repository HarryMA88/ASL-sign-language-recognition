import torch.nn as nn
import torchvision.models as models
from ml.config import TRAIN_CONFIG

def get_model(num_classes=TRAIN_CONFIG["num_classes"]):
    """
    Creates a modified Resnet model for 28×28 grayscale images.

    Args:
        num_classes (int): Number of output classes for the model.
    """
    model = models.resnet18(weights=None)

    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.bn1 = nn.BatchNorm2d(64)
    model.relu = nn.ReLU(inplace=True)
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
