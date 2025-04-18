import numpy as np
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from backend.dataset import SignLanguageDataset

class DatasetLoader(QThread):
    batchReady     = pyqtSignal(list)   # emits list of (QPixmap, label)
    datasetLoaded  = pyqtSignal(object) # emits the dataset object
    finished       = pyqtSignal()

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        # load dataset (this may take time)
        ds = SignLanguageDataset.from_csv(self.path, transform=None)
        self.datasetLoaded.emit(ds)

        batch = []
        for img, lbl in ds:
            # convert to QPixmap thumbnail
            img_np = img.numpy().squeeze() if isinstance(img, torch.Tensor) else img
            qimg   = QImage((img_np * 255).astype(np.uint8), 28, 28, 28, QImage.Format_Grayscale8)
            pix    = QPixmap.fromImage(qimg).scaled(64, 64, Qt.KeepAspectRatio)
            batch.append((pix, lbl))

            if len(batch) >= 50:
                self.batchReady.emit(batch)
                batch.clear()

        # emit any remainder
        if batch:
            self.batchReady.emit(batch)

        self.finished.emit()
