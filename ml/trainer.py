import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.amp import autocast, GradScaler
from copy import deepcopy
import numpy as np

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


def train_loop(model, train_loader, val_loader, epochs,
               emit_epoch=None, should_stop=None, model_name="model"):
    device = next(model.parameters()).device

    # ─── set up criterion, optimizer, scheduler ─────────────────────────────
    if model_name.lower() == "alexnet":
        # label smoothing
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # grab features & head modules dynamically
        base = getattr(model, "backbone", model)
        feats = base.features.parameters()
        head  = base.classifier.parameters()

        # discriminative learning rates
        optimizer = Adam(
            [
                {"params": feats, "lr": TRAIN_CONFIG["learning_rate"] * 1e-3},  # e.g. 1e-5
                {"params": head,  "lr": TRAIN_CONFIG["learning_rate"] * 1e-2},  # e.g. 1e-4
            ],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        # more conservative schedule for fine‑tuning
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

        # freeze all convs for first few epochs
        for p in base.features.parameters():
            p.requires_grad = False

    else:
        # unchanged for other models
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    scaler = GradScaler(device=TRAIN_CONFIG["device"])
    metrics = {"train_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(epochs):
        if should_stop and should_stop():
            break

        # after epoch 5, unfreeze all AlexNet convs
        if epoch == 5 and model_name.lower() == "alexnet":
            for p in base.features.parameters():
                p.requires_grad = True

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            if should_stop and should_stop():
                break

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # MixUp only for AlexNet
            if model_name.lower() == "alexnet":
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(images.size(0))
                mixed = lam * images + (1 - lam) * images[idx]
                labels_a, labels_b = labels, labels[idx]
                inputs = mixed
            else:
                inputs = images

            with autocast(device_type="cuda"):
                outputs = model(inputs)
                if model_name.lower() == "alexnet":
                    loss = (
                        lam * criterion(outputs, labels_a)
                        + (1 - lam) * criterion(outputs, labels_b)
                    )
                else:
                    loss = criterion(outputs, labels)

            # backprop with gradient clipping
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total if total else 0.0
        metrics["train_loss"].append(train_loss)
        scheduler.step()

        # ─── validation ────────────────────────────────────────────────────────
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                if should_stop and should_stop():
                    break
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total else 0.0
        metrics["val_acc"].append(val_acc)

        if emit_epoch:
            elapsed = time.time() - start_time
            emit_epoch(epoch + 1, train_loss, val_acc, elapsed)

    # ─── save checkpoint ────────────────────────────────────────────────────
    os.makedirs("saved_models", exist_ok=True)
    ts = time.strftime("%m%d_%H%M")
    path = os.path.join("saved_models", f"{model_name}_{ts}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "metadata": {
            "model_choice": model_name.lower(),
            "input_shape": (28, 28),
            "num_classes": TRAIN_CONFIG["num_classes"]
        }
    }, path)

    metrics["elapsed"] = time.time() - start_time
    return metrics