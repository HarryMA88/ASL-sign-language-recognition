import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from backend.dataset import SignLanguageDataset
from ml.models import model_registry
from ml.config import TRAIN_CONFIG

# === CONFIG ===
MODEL_NAME = "lebron"
MODEL_FILE = "saved_models/lebron_20250417_190754.pt"
CSV_PATH = r"C:\Users\Dhruv\Downloads\sign_mnist_digits\sign_mnist_alpha_digits_test.csv"

# === TRANSFORM ===
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((TRAIN_CONFIG["normalize_mean"],), (TRAIN_CONFIG["normalize_std"],))
])

# === DATASET ===
test_dataset = SignLanguageDataset.from_csv(CSV_PATH, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

# === MODEL LOAD ===
device = torch.device(TRAIN_CONFIG["device"])
model_fn = model_registry[MODEL_NAME.lower()]
model = model_fn(num_classes=TRAIN_CONFIG["num_classes"])
checkpoint = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# === EVALUATION ===
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
print(f"\nMisclassified Samples (up to 10 shown):")

for i, (img, true, pred) in enumerate(misclassified[:10]):
    plt.imshow(img.squeeze(), cmap="gray")
    plt.title(f"True: {true} | Pred: {pred}")
    plt.axis("off")
    plt.show()
