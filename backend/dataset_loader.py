import numpy as np
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from backend.dataset import SignLanguageDataset

# This class is for a background thread for loading the dataset images so that the gui doesn't freeze
class DatasetLoader(QThread):
    # This is for displaying the images to the gui in batches to improve performance
    batchReady = pyqtSignal(list)
    # This is for displaying the full dataset with the count of each sign once its loaded
    datasetLoaded = pyqtSignal(object)
    # This is a signal for when we are finished loading the dataset
    finished = pyqtSignal()

    # This is the constructor and takes the file path for the csv to load and stores it to be loaded
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    # This is where the main logic of the thread is
    def run(self):
        # This loads the dataset from a csv file and then displays it to the gui
        ds = SignLanguageDataset.from_csv(self.path, transform=None)
        self.datasetLoaded.emit(ds)

        # We make a batch so we can load the dataset in batches instead of all of the images at once to improve performance
        batch = []
        for img, lbl in ds:
            # This converts the dataset into thumbnails
            img_np = img.numpy().squeeze() if isinstance(img, torch.Tensor) else img
            qimg   = QImage((img_np * 255).astype(np.uint8), 28, 28, 28, QImage.Format_Grayscale8)
            pix    = QPixmap.fromImage(qimg).scaled(64, 64, Qt.KeepAspectRatio)
            batch.append((pix, lbl))

            # This loads every 50 thumbnails at once
            if len(batch) >= 50:
                self.batchReady.emit(batch)
                batch.clear()

        # This loads the remaining thumbnails
        if batch:
            self.batchReady.emit(batch)

        self.finished.emit()
