import numpy as np
import torch

from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap

from backend.webcam_thread import WebcamThread
from ml.models import model_registry

class PredictionTab(QWidget):
    """
    Tab where the user loads a saved .pt and then clicks thumbnails
    in the Dataset Viewer to see an enlarged image + live prediction.
    """
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)

        # — Load Model Button —
        self.btn_load = QPushButton("Load Model…")
        self.btn_load.clicked.connect(self._on_load_model)
        self.layout.addWidget(self.btn_load)

        # — Display an enlarged image here —
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.img_label)

        # — Display the predicted label —
        self.pred_label = QLabel("Prediction: N/A")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.pred_label)

        # Internal state
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick Model", "", "PyTorch Checkpoint (*.pt)"
        )
        if path:
            self._load_model(path)

    def _load_model(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device)
            meta = ckpt.get("metadata", {})
            state = ckpt.get("model_state", ckpt)

            choice = meta.get("model_choice", "").lower()
            input_shape = tuple(meta.get("input_shape", (28, 28)))
            num_classes = meta.get("num_classes", 36)

            ModelClass = model_registry.get(choice)
            if ModelClass is None:
                QMessageBox.warning(
                    self, "Unknown Model",
                    f"Model '{choice}' not recognized."
                )
                return

            model = ModelClass(input_shape, num_classes)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()

            self.model = model
            QMessageBox.information(
                self, "Loaded",
                "Model loaded successfully and ready to predict."
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load model:\n{e}"
            )

    @pyqtSlot(np.ndarray)
    def predict_image(self, img: np.ndarray):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return

        # Show the image enlarged
        h, w = img.shape
        qimg = QImage(
            (img * 255).astype("uint8").data,
            w, h, w,
            QImage.Format_Grayscale8
        )
        pix = QPixmap.fromImage(qimg).scaled(
            200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.img_label.setPixmap(pix)

        # Run inference
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(tensor)
            idx = int(out.argmax(1).item())

        # Map back to label
        if idx < 26:
            label = chr(idx + ord("A"))
        else:
            label = str(idx - 26)
        self.pred_label.setText(f"Prediction: {label}")
