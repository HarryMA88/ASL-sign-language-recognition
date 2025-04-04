import torch
from torch import nn
from tqdm import tqdm
import os

def train_model(model, train_loader, val_loader, device, config):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config["epochs"]
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate_model(model, val_loader, device)

        print(f"Epoch {epoch+1}: Loss = {running_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, config)

    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")


def evaluate_model(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def save_model(model, config):
    path = config.get("save_path", "models")
    os.makedirs(path, exist_ok=True)
    name = config["name"]
    full_path = os.path.join(path, f"{name}_best.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Best model saved at: {full_path}")
