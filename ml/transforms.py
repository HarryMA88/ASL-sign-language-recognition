from torchvision import transforms
from ml.config import TRAIN_CONFIG

def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        # slight rotation
        transforms.RandomRotation(10),
        # small affine: tiny shift, scale, shear
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5
        ),
        # horizontal flip half the time
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (TRAIN_CONFIG["normalize_mean"],),
            (TRAIN_CONFIG["normalize_std"],)
        ),
    ])


def get_test_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))
    ])
