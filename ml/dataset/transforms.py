from torchvision import transforms
from PIL import Image

class GrayscaleToRGB:
    def __call__(self, img: Image.Image):
        return img.convert("RGB")

def get_transforms(train: bool):
    base = [
        transforms.Resize((224, 224)),
        GrayscaleToRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ]

    if train:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]
        return transforms.Compose(aug + base)

    return transforms.Compose(base)
