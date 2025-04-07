import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class ASLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pixels = row.drop("label").values.astype("uint8").reshape(28, 28)
        image = Image.fromarray(pixels)
        image = self.transform(image)
        label = row["label"]
        return image, label