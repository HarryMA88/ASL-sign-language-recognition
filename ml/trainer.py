import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from backend.dataset import SignLanguageDataset
from ml.config import TRAIN_CONFIG
from ml.transforms import get_train_transforms, get_test_transforms
from typing import Callable



def to_device(*args, device):
    """
    Moves the given arguments to the specified device (CPU or GPU).
    """
    return [x.to(device) for x in args]

def make_loaders(dataset: SignLanguageDataset, train_pct: float, batch_size: int):
    """
    Splits a dataset into training and validation sets, applies the appropriate transforms, 
    and returns their DataLoaders.

    Args:
        dataset (SignLanguageDataset): The dataset represented by a custom class.
        train_pct (float): The proportion of the dataset to use for training (0.0 to 1.0).
        batch_size (int): The number of samples per batch.

    Returns:
        tuple: DataLoaders for training and validation sets.
    """
    val_size = int((1 - train_pct) * len(dataset))
    train_size = len(dataset) - val_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_subset.dataset.transform = get_train_transforms()
    val_subset.dataset.transform = get_test_transforms()

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, val_loader


def train_loop(model :nn.Module, train_loader :DataLoader, val_loader :DataLoader, epochs :int, emit_epoch :Callable[[int, float, float, float], None], should_stop :Callable[[], bool], model_name :str="model"):
    """
    Runs a training loop with the capacity to stop and returns epoch to epoch callbacks. Saves the model to disk 
    after training is completeed. The training loop makes use of PyTorch's Automatic Mixed Precision (AMP) to 
    improve training performance on CUDA-enabled devices via `torch.amp.autocast` and `GradScaler`.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        epochs (int): Total number of training epochs.
        emit_epoch (callable): Hook to emit progress after each epoch.
        should_stop (callable): Hook to stop training early.
        model_name (str): Filename prefix for the saved model.

    Returns:
        dict: Training metrics including loss, val accuracy, and elapsed time.
    """
    device = next(model.parameters()).device # Automatically detect which device the model is on (CPU or CUDA)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(device=TRAIN_CONFIG["device"]) # Used for automatic mixed precision (AMP) to speed up training on CUDA

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

            images, labels = to_device(images, labels, device=device)
            optimizer.zero_grad()

            if(model_name.lower() == "alexnet"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            else:
                # Enable automatic mixed precision for faster training with float16 on CUDA
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
        metrics["train_loss"].append(train_loss)
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                if should_stop():
                    break
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        metrics["val_acc"].append(val_acc)
        emit_epoch(epoch + 1, train_loss, val_acc, time.time() - start_time)

    os.makedirs("saved_models", exist_ok=True)
    timestamp = time.strftime("%m%d_%H%M")
    save_path = os.path.join("saved_models", f"{model_name}_{timestamp}.pt")
    torch.save({
        "model_state": model.state_dict(),
        "metadata": {
            "model_choice": model_name.lower(),
            "input_shape": (28, 28),
            "num_classes": TRAIN_CONFIG["num_classes"]
        }
    }, save_path)

    metrics["elapsed"] = time.time() - start_time
    return metrics