import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import torch.amp
from transforms import get_test_transforms, get_train_transforms
from config import TRAIN_CONFIG
from dataset import ASLDataset
from models import model_registry
import matplotlib.pyplot as plt


def prepare_data_loaders():
    full_dataset = ASLDataset("data/sign_mnist_alpha_digits_train.csv", transform=get_train_transforms())
    test_set = ASLDataset("data/sign_mnist_alpha_digits_test.csv", transform=get_test_transforms())

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True,
                              pin_memory=True, num_workers=2, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                            pin_memory=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False,
                             pin_memory=True, num_workers=2, persistent_workers=True)

    return train_loader, val_loader, test_loader


def run_training(model, train_loader, val_loader, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"],
                           weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    scaler = torch.amp.GradScaler()

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        # Validation phase
        model.eval()
        correct, total, val_loss = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss /= total
        scheduler.step()

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} "
              f"| Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f"{model_name}_model.pt")
    print(f"Model saved as {model_name}_model.pt")


def run_test(model, test_loader, device, model_name):
    model.load_state_dict(torch.load(f"{model_name}_model.pt", map_location=device, weights_only=True))
    model.eval()

    correct, total = 0, 0
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((images[i].cpu(), labels[i].item(), preds[i].item()))

    print(f"Test Accuracy: {correct / total:.4f}")

    print(f"\nMisclassified Samples (showing up to 50):")
    for i, (img, true_label, pred_label) in enumerate(misclassified[:50]):
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(f"True: {true_label} | Pred: {pred_label}")
        plt.axis("off")
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--model", type=str, default="resnet", choices=model_registry.keys())
    args = parser.parse_args()

    device = torch.device(TRAIN_CONFIG["device"])
    model = model_registry[args.model](num_classes=TRAIN_CONFIG["num_classes"]).to(device)

    train_loader, val_loader, test_loader = prepare_data_loaders()

    if args.mode == "train":
        run_training(model, train_loader, val_loader, device, args.model)
    elif args.mode == "test":
        run_test(model, test_loader, device, args.model)


if __name__ == "__main__":
    main()
