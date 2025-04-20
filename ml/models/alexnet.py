import torch.nn as nn
from torchvision.models import alexnet
from ml.config import TRAIN_CONFIG

def get_model(num_classes: int = TRAIN_CONFIG["num_classes"]):
    """
    Creates a modified AlexNet model for 28×28 grayscale images.
    Boosted with additional classifier capacity, BatchNorm, and Dropout.
    """
    model = alexnet(weights=None)

    # 1. Modify input conv layer for grayscale input
    model.features[0] = nn.Conv2d(
        in_channels=1, out_channels=64,
        kernel_size=3, stride=1, padding=1
    )

    # 2. Use larger spatial pooling output for better representation
    model.avgpool = nn.AdaptiveAvgPool2d((3, 3))  # 256 × 3 × 3

    # 3. Enhanced classifier with norm, dropout
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 3 * 3, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes)
    )

    # 4. Safe weight initialization
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)
    return model
