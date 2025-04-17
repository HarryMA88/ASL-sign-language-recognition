import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from copy import deepcopy

from ml.config import TRAIN_CONFIG
from ml.transforms import get_train_transforms, get_test_transforms


def make_loaders(dataset, train_pct, batch_size):
    val_size = int((1 - train_pct) * len(dataset))
    train_size = len(dataset) - val_size

    train_ds = deepcopy(dataset)
    val_ds = deepcopy(dataset)
    train_ds.transform = get_train_transforms()
    val_ds.transform = get_test_transforms()

    train_subset, _ = random_split(train_ds, [train_size, val_size])
    _, val_subset = random_split(val_ds, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def train_loop(model, train_loader, val_loader, epochs, emit_epoch=None, should_stop=None, model_name="model"):
    device = next(model.parameters()).device

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(device=TRAIN_CONFIG["device"])

    metrics = {"train_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(epochs):
        if should_stop and should_stop():
            break

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            if should_stop and should_stop():
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        metrics["train_loss"].append(train_loss)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                if should_stop and should_stop():
                    break
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        metrics["val_acc"].append(val_acc)

        if emit_epoch:
            emit_epoch(epoch + 1, train_loss, val_acc, time.time() - start_time)

    os.makedirs("saved_models", exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M")
    save_path = os.path.join("saved_models", f"{model_name}_{timestamp}.pt")
    torch.save(model.state_dict(), save_path)

    metrics["elapsed"] = time.time() - start_time
    return metrics
