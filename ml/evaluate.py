from models.lebron2_3.lebron import Lebron23
import torch
import pandas as pd

# Load model
model = Lebron23(num_classes=36)
model.load_state_dict(torch.load("saved_models/lebron2_3_best.pt"))
model.eval()

# Load test data
df = pd.read_csv("data/sign_mnist_alpha_digits_test.csv")

correct = 0
total = len(df)

for i in range(total):
    label = df.iloc[i, 0]
    pixels = df.iloc[i, 1:].values.astype("float32") / 255.0
    x = torch.tensor(pixels).view(1, 1, 28, 28)

    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()

    if pred == label:
        correct += 1
    else:
        print(f"Mismatch at sample {i}: Predicted = {pred}, Actual = {label}")

print(f"\n Final Test Accuracy: {correct / total:.4f} ({correct}/{total})")
