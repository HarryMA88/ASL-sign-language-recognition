import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        lbl = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, lbl

    @classmethod
    def from_csv(cls, path, transform=None):
        df = pd.read_csv(path)
        labels = df.iloc[:, 0].values
        images = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype("uint8")  # important: keep it uint8 for PIL
        return cls(images, labels, transform=transform)
