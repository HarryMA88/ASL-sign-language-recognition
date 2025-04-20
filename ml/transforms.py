from torchvision import transforms
from ml.config import TRAIN_CONFIG

def get_train_transforms():
    """
    Applies data augmentation to 28×28 grayscale images for training.

    Transforms:
    - Random rotation, affine distortion, and flip
    - Converts to tensor and normalizes using standardised numbers from TRAIN_CONFIG

    Returns:
    torchvision.transforms.Compose: A callable transform pipeline for training images.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=5
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (TRAIN_CONFIG["normalize_mean"],),
            (TRAIN_CONFIG["normalize_std"],)
        ),
    ])


def get_test_transforms():
    """
    Applies transforms to 28×28 grayscale images for testing.
    
    Transforms:
    - Converts to tensor and normalizes using standardised numbers from TRAIN_CONFIG

    Returns:
    torchvision.transforms.Compose: A callable transform pipeline for testing images.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))
    ])
