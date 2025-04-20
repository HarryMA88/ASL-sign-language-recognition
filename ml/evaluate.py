import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from backend.dataset import SignLanguageDataset
from ml.models import model_registry
from ml.config import TRAIN_CONFIG
from ml.transforms import get_test_transforms

# === CONFIG ===
#Change the model name to the one you want to evaluate. Change the path to the model file.
MODEL_NAME = "resnet" #Change me
MODEL_FILE = "saved_models/resnet_0420_1856.pt" #Change me

CSV_PATH   = "datasets/sign_mnist_alpha_digits_test.csv"

# === DATASET ===
test_dataset = SignLanguageDataset.from_csv(CSV_PATH, transform= get_test_transforms())
test_loader  = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False)

# === MODEL LOAD ===
device = torch.device(TRAIN_CONFIG["device"])
model_fn = model_registry[MODEL_NAME.lower()]
model = model_fn(num_classes=TRAIN_CONFIG["num_classes"]).to(device)

checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# === EVALUATION ===
correct, total = 0, 0
misclassified = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        for i in range(len(labels)):
            if preds[i] != labels[i]:
                misclassified.append((images[i].cpu(), labels[i].item(), preds[i].item()))

# === RESULTS ===
print(f"Test Accuracy: {correct / total:.4f}")