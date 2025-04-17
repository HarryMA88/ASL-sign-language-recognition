import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import torch.amp
import os
from ml.transforms import get_test_transforms, get_train_transforms
from ml.config import TRAIN_CONFIG
from backend.dataset import SignLanguageDataset
from ml.models import model_registry



class ToTensorNormalize:
    def __call__(self, image):
        image = torch.tensor(image).unsqueeze(0) / 255.0
        return transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))(image)

def prepare_data_loaders():
    full_train_set = SignLanguageDataset("data/sign_mnist_alpha_digits_train.csv", transform=get_train_transforms)
    test_set = SignLanguageDataset("data/sign_mnist_alpha_digits_test.csv", transform=get_test_transforms)

    val_size = int(0.2 * len(full_train_set))
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, pin_memory=True, pin_memory_device="cuda")
    val_loader = DataLoader(val_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, pin_memory=True, pin_memory_device="cuda")
    test_loader = DataLoader(test_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, pin_memory=True, pin_memory_device="cuda")

    return train_loader, val_loader, test_loader



def run_training(model, train_loader, device, model_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG["epochs"])
    scaler = torch.amp.GradScaler(device=TRAIN_CONFIG["device"])

    model.eval()
    val_correct, val_total, val_loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    val_loss = val_loss / val_total
    print(f"[Before Training] Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device="cuda"):
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
        scheduler.step()

        # Post-epoch validation
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), f"{model_name}_model.pt")
    print("Model saved as best_model.pt")


def run_test(model, test_loader, device, model_path):
    import matplotlib.pyplot as plt

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
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
    parser.add_argument("--model_name", type=str, required=True, choices=model_registry.keys())
    parser.add_argument("--model_file", type=str, required=True, help="Path to the .pt file in saved_models/")
    args = parser.parse_args()

    model_path = os.path.join("saved_models", args.model_file)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at: {model_path}")
        return

    device = torch.device(TRAIN_CONFIG["device"])
    model_fn = model_registry[args.model_name]
    model = model_fn(num_classes=TRAIN_CONFIG["num_classes"]).to(device)

    test_loader = prepare_data_loaders()
    run_test(model, test_loader, device, model_path)

if __name__ == "__main__":
    main()