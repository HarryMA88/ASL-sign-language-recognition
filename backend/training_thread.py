import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PyQt5.QtCore import QThread, pyqtSignal
from ml.config import TRAIN_CONFIG
from ml.models import model_registry
from ml.transforms import get_train_transforms
from ml.dataset import ASLDataset

class TrainingThread(QThread):
    epoch_signal    = pyqtSignal(int, float, float, float)
    finished_signal = pyqtSignal(dict)
    error_signal    = pyqtSignal(str)

    def __init__(self, model_choice, dataset, train_pct, batch_size, num_epochs, device):
        super().__init__()
        self.model_choice    = model_choice
        self.dataset         = dataset
        self.train_pct       = train_pct
        self.batch_size      = batch_size
        self.num_epochs      = num_epochs
        self.device          = device
        self._is_running     = True
        self.metadata        = {}

    def run(self):
        try:
            # data loaders
            loader = ASLDataset(self.dataset, transform=get_train_transforms())
            train_size = int(self.train_pct * len(loader))
            val_size   = len(loader) - train_size
            train_set, val_set = random_split(loader, [train_size, val_size])
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader   = DataLoader(val_set,   batch_size=self.batch_size, shuffle=False)

            # build model
            ModelClass = model_registry[self.model_choice.lower()]
            model = ModelClass(num_classes=TRAIN_CONFIG["num_classes"])
            model.to(self.device)

            opt       = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
            crit      = nn.CrossEntropyLoss()
            start     = time.time()
            metrics   = {"train_loss": [], "val_acc": []}

            for e in range(1, self.num_epochs+1):
                if not self._is_running: break
                model.train()
                running = 0
                for X, y in train_loader:
                    if not self._is_running: break
                    X, y = X.to(self.device), y.to(self.device)
                    opt.zero_grad()
                    out = model(X)
                    loss= crit(out, y)
                    loss.backward()
                    opt.step()
                    running += loss.item()*X.size(0)
                train_loss = running/len(train_loader.dataset)
                metrics["train_loss"].append(train_loss)

                # validation
                model.eval()
                correct = total = 0
                with torch.no_grad():
                    for X, y in val_loader:
                        X, y = X.to(self.device), y.to(self.device)
                        preds = model(X).argmax(1)
                        correct += (preds==y).sum().item()
                        total   += y.size(0)
                val_acc = correct/total if total else 0
                metrics["val_acc"].append(val_acc)

                elapsed = time.time() - start
                self.epoch_signal.emit(e, train_loss, val_acc, elapsed)

            self.finished_signal.emit(metrics)
            self.metadata = {
                "model_choice": self.model_choice,
                "train_pct": self.train_pct,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                **metrics,
                "elapsed": elapsed
            }
            self.model = model

        except Exception as ex:
            self.error_signal.emit(str(ex))

    def stop(self):
        self._is_running = False