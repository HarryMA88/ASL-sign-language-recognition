import torch.nn as nn
from torchvision import models

def build_model(num_classes: int) -> nn.Module:
    model = models.alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
