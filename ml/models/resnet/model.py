import torch.nn as nn
from torchvision import models

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model