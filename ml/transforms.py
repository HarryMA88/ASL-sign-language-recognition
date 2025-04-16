from torchvision import transforms
from config import TRAIN_CONFIG

def get_train_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((TRAIN_CONFIG["resize"], TRAIN_CONFIG["resize"])),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))
    ])

def get_test_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((TRAIN_CONFIG["resize"], TRAIN_CONFIG["resize"])),
        transforms.ToTensor(),
        transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))
    ])
