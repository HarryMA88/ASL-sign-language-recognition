import torch
import pandas as pd
from torch.utils.data import Dataset

class SignLanguageDataset(Dataset):
    """
    This is a class for a dataset that is specifically for sign language and defines methods specifically for a sign language dataset
    """
    def __init__(self, images, labels, transform=None):
        """
        This is the constructor method and stores the images and corresponding labels for each image and which transform to apply to the images
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        This is a helper method to get the number of images in the dataset
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        This method returns the image and its corresponding label at the indexed position in the dataset
        """
        img = self.images[idx]
        lbl = int(self.labels[idx])

        # This applies the transform specified in the constructor to the image, or a default transformation to the image if none was specified
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0

        return img, lbl

    @classmethod
    def from_csv(cls, path, transform=None):
        """
        This method converts a dataset in the csv format to this class
        """
        df = pd.read_csv(path)
        labels = df.iloc[:, 0].values
        images = df.iloc[:, 1:].values.reshape(-1, 28, 28).astype("uint8")  # important: keep it uint8 for PIL
        return cls(images, labels, transform=transform)
