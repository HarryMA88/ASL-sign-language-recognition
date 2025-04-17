import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from ml.config import TRAIN_CONFIG


def make_loaders(dataset, train_pct, batch_size):
    val_size = int((1 - train_pct) * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True),
    )


def train_loop(model, train_loader, val_loader, epochs, emit_epoch=None, should_stop=None, model_name="model"):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    start = time.time()
    metrics = {"train_loss": [], "val_acc": []}

    for ep in range(1, epochs + 1):
        if should_stop and should_stop():
            break

        model.train()
        running_loss = 0.0
        correct = total = 0
        for x, y in train_loader:
            if should_stop and should_stop():
                break

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total if total else 0.0
        metrics["train_loss"].append(train_loss)
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                if should_stop and should_stop():
                    break
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total if total else 0.0
        metrics["val_acc"].append(val_acc)

        elapsed = time.time() - start
        if emit_epoch:
            emit_epoch(ep, train_loss, val_acc, elapsed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("saved_models", exist_ok=True)
    save_path = os.path.join("saved_models", f"{model_name}_{timestamp}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "metadata": {"timestamp": timestamp, "epochs": epochs, **metrics}
    }, save_path)
    print(f"Model saved to {save_path}")

    return {
        "train_loss": metrics["train_loss"],
        "val_acc": metrics["val_acc"],
        "elapsed": time.time() - start
    }
