import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image

train_transforms = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])


class SignLanguageDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.labels = df.iloc[:, 0].values.astype(np.int64)
        self.images = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype(np.uint8)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)  # Apply torchvision transforms on PIL image
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_loaders(csv_path, batch_size=64, val_split=0.2):
    full_dataset = SignLanguageDataset(csv_path, transform=train_transforms)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Apply test_transforms to val_set
    val_set.dataset.transform = test_transforms

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    return train_loader, val_loader