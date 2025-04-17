import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from ml.config import TRAIN_CONFIG
from ml.dataset import ASLDataset
from ml.transforms import get_train_transforms

def make_loaders(csv_path, train_pct, batch_size):
    full = ASLDataset(csv_path, transform=get_train_transforms())
    val_size = int((1-train_pct) * len(full))
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size])
    return (
      DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True),
      DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True),
    )

def train_loop(model, train_loader, val_loader, device, epochs, emit_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(),
                     lr=TRAIN_CONFIG["learning_rate"],
                     weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler    = GradScaler()

    start = time.time()
    for ep in range(1, epochs+1):
        model.train()
        running_loss = 0.
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds==y).sum().item()
                total   += y.size(0)
        val_acc = correct/total
        elapsed = time.time() - start

        emit_epoch(ep, train_loss, val_acc, elapsed)

    return {
      "train_loss": None,  # you can store lists if you like
      "val_acc":    None,
      "elapsed":    elapsed,
    }
