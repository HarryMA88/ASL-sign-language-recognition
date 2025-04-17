import torch
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    """
    Simple PyTorch Dataset wrapping NumPy arrays of grayscale images and integer labels.
    """
    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        lbl = int(self.labels[idx])
        return img, lbl