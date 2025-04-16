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



class ToTensorNormalize:
    def __call__(self, image):
        image = torch.tensor(image).unsqueeze(0) / 255.0
        return transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))(image)

def prepare_data_loaders():
    transform = ToTensorNormalize()

    train_set = ASLDataset("data/sign_mnist_alpha_digits_train.csv", transform=transform)
    test_set = ASLDataset("data/sign_mnist_alpha_digits_test.csv", transform=transform)

    train_loader = DataLoader(train_set, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, pin_memory=True, pin_memory_device="cuda")
    val_loader = None  # Not used anymore
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


    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
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

        scheduler.step()
        train_acc = correct / total
        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Loss: {running_loss / total:.4f}")

    torch.save(model.state_dict(), f"{model_name}_model.pt")
    print("Model saved as best_model.pt")



def run_test(model, test_loader, device, model_name):
    import matplotlib.pyplot as plt

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

            # Track mismatches
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    misclassified.append((images[i].cpu(), labels[i].item(), preds[i].item()))

    print(f"Test Accuracy: {correct / total:.4f}")

    # Show up to 5 misclassified samples
    print("\nMisclassified Samples (up to 5 shown):")
    for i, (img, true_label, pred_label) in enumerate(misclassified[:5]):
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
    model_fn = model_registry[args.model]
    model = model_fn(num_classes=TRAIN_CONFIG["num_classes"]).to(device)

    train_loader, val_loader, test_loader = prepare_data_loaders()

    if args.mode == "train":
        run_training(model, train_loader, device, args.model)
    elif args.mode == "test":
        run_test(model, test_loader, device, args.model)

if __name__ == "__main__":
    main()
