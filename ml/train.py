import pandas as pd
from torch.utils.data import DataLoader
from config import NUM_CLASSES, DEVICE, SAVE_PATH
from models import get_model
from dataset.loader import ASLDataset
from dataset.transforms import get_transforms
from trainer.trainer import train


def main():
    # Load CSV
    df = pd.read_csv("data/sign_mnist_alpha_digits_train.csv")  # adjust path if needed

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    # Data loaders
    train_loader = DataLoader(
        ASLDataset(train_df, get_transforms(train=True)),
        batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        ASLDataset(val_df, get_transforms(train=False)),
        batch_size=64, num_workers=4, pin_memory=True
    )

    # Hyperparams (passed in or set dynamically)
    model_name = "resnet"  # or "alexnet"
    epochs = 30
    lr = 1e-3

    # Init model
    model = get_model(model_name, NUM_CLASSES)

    # Train
    train(model, train_loader, val_loader, epochs, lr, model_name)

if __name__ == "__main__":
    main()