import torch
import pandas as pd
from torchvision import transforms
from config import DEVICE, NUM_CLASSES, SAVE_PATH
from models import get_model

# === CONFIG ===
MODEL_NAME = "resnet"  # or "alexnet"
MODEL_PATH = f"{SAVE_PATH}/{MODEL_NAME}_best.pt"
CSV_PATH = "data/sign_mnist_alpha_digits_test.csv"

# === MODEL SETUP ===
model = get_model(MODEL_NAME, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# === LOAD DATA ===
df = pd.read_csv(CSV_PATH)
correct = 0
total = len(df)

for i in range(total):
    label = int(df.iloc[i, 0])
    pixels = df.iloc[i, 1:].values.astype("uint8").reshape(28, 28)
    image = transform(pixels)
    x = image.unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()

    if pred == label:
        correct += 1
    else:
        print(f"Mismatch at sample {i}: Predicted = {pred}, Actual = {label}")

print(f"\nFinal Test Accuracy: {correct / total:.4f} ({correct}/{total})")
