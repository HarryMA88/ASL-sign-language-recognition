import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet, AlexNet_Weights
from ml.config import TRAIN_CONFIG

class AlexNet28(nn.Module):
    def __init__(self, num_classes: int = TRAIN_CONFIG["num_classes"]):
        super().__init__()
        # 1) load pretrained ImageNet AlexNet (conv1 stays 3→64)
        self.backbone = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        # 2) replace the classifier head only
        #    original: [Dropout, Linear(256*6*6→4096), ReLU, Dropout, Linear(4096→4096), ReLU, Linear(4096→1000)]
        #    new:      [Dropout, Linear(256*3*3→1024), ReLU, Dropout, Linear(1024→512), ReLU, Linear(512→num_classes)]
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x: B×1×28×28 from your existing transforms
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = x.repeat(1, 3, 1, 1)             # make it B×3×224×224
        return self.backbone(x)

def get_model(num_classes=TRAIN_CONFIG["num_classes"]):
    return AlexNet28(num_classes)
