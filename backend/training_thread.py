from PyQt5.QtCore import QThread, pyqtSignal
import torch

from ml.trainer import make_loaders, train_loop
from ml.models import model_registry
from ml.config import TRAIN_CONFIG
from backend.dataset import SignLanguageDataset


class TrainingThread(QThread):
    epoch_signal = pyqtSignal(int, float, float, float)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, model_choice: str, dataset: SignLanguageDataset, train_pct: float, batch_size: int, num_epochs: int):
        super().__init__()
        self.model_choice = model_choice.lower()
        self.dataset = dataset
        self.train_pct = train_pct
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_running = True
        self.metadata = {}

    def run(self):
        try:
            train_loader, val_loader = make_loaders(self.dataset, self.train_pct, self.batch_size)
            model_fn = model_registry[self.model_choice]
            model = model_fn(num_classes=TRAIN_CONFIG["num_classes"])
            model.to(self.device)

            metrics = train_loop(
                model,
                train_loader,
                val_loader,
                epochs=self.num_epochs,
                emit_epoch=self.epoch_signal.emit,
                should_stop=lambda: not self._is_running,
                model_name=self.model_choice
            )

            self.metadata = {
                "model_choice": self.model_choice,
                "train_pct": self.train_pct,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                **metrics
            }
            self.model = model
            self.finished_signal.emit(metrics)

        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False
