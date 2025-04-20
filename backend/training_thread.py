from PyQt5.QtCore import QThread, pyqtSignal
import torch

from ml.trainer import make_loaders, train_loop
from ml.models import model_registry
from ml.config import TRAIN_CONFIG
from backend.dataset import SignLanguageDataset

class TrainingThread(QThread):
    """
    This class is a background thread for training the models
    """
    # This is to display how many epochs have passed
    epoch_signal = pyqtSignal(int, float, float, float)
    # This is to display the metrics of the training once finished
    finished_signal = pyqtSignal(dict)
    # This is to display any potential errors
    error_signal = pyqtSignal(str)

    def __init__(self, model_choice: str, dataset: SignLanguageDataset, train_pct: float, batch_size: int, num_epochs: int):
        """
        This is the constructor which takes in the settings for which we are going to train our models with
        """
        super().__init__()
        self.model_choice = model_choice.lower()
        self.dataset = dataset
        self.train_pct = train_pct
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        # Enables cuda cores for training if available to the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_running = True
        self.metadata = {}

    def run(self):
        """
        This is where the main logic of the thread is in
        """
        # Tries to train a model with the specified settings otherwise, throw an exception
        try:
            # Splits the dataset into a training set and testing set
            train_loader, val_loader = make_loaders(self.dataset, self.train_pct, self.batch_size)

            # Picks out the specified model
            model_fn = model_registry[self.model_choice]
            model = model_fn(num_classes=TRAIN_CONFIG["num_classes"])
            model.to(self.device)

            # Starts a loop for training the model
            metrics = train_loop(
                model,
                train_loader,
                val_loader,
                epochs=self.num_epochs,
                emit_epoch=self.epoch_signal.emit,
                should_stop=lambda: not self._is_running,
                model_name=self.model_choice
            )

            # Stores metadata about the model
            self.metadata = {
                "model_choice": self.model_choice,
                "train_pct": self.train_pct,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                **metrics
            }
            self.model = model
            # Displays the metrics of training the model
            self.finished_signal.emit(metrics)

        # Throws exception if training the model is unsuccessful
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        """
        This method stops training the model
        """
        self._is_running = False
