import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import SAVE_PATH, DEVICE

def save_model(model, name):
    os.makedirs(SAVE_PATH, exist_ok=True)
    full_path = os.path.join(SAVE_PATH, f"{name}_best.pt")
    torch.save(model.state_dict(), full_path)
    print(f"Best model saved at: {full_path}")

def train(model, train_loader, val_loader, num_epochs, lr, model_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        total, correct = 0, 0
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        val_acc = evaluate(model, val_loader, DEVICE)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
        
    if best_model_state:
            torch.save(best_model_state, os.path.join(SAVE_PATH, f"{model_name}_best.pt"))
            print(f"Best model saved after training. Val Acc: {best_acc:.4f}")

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0
