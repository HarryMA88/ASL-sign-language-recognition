import numpy as np
import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QScrollArea, QGridLayout
)
from PyQt5.QtGui import QImage, QPixmap

from ml.models import model_registry
from backend.dataset import SignLanguageDataset
from frontend.prediction_popup import PredictionPopup
from frontend.webcam_popup import WebcamPopup



class PredictionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layout = QVBoxLayout(self)
        self.control_bar = QHBoxLayout()
        self.grid_area = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)

        self.btn_model = QPushButton("Select Model")
        self.btn_model.clicked.connect(self.select_model)
        self.btn_data = QPushButton("Select Dataset")
        self.btn_data.clicked.connect(self.select_dataset)
        self.btn_webcam = QPushButton("Open Webcam")
        self.btn_webcam.clicked.connect(self.open_webcam)
        self.control_bar.addWidget(self.btn_webcam)

        self.control_bar.addWidget(self.btn_model)
        self.control_bar.addWidget(self.btn_data)
        self.layout.addLayout(self.control_bar)

        self.grid_area.setWidget(self.grid_widget)
        self.grid_area.setWidgetResizable(True)
        self.layout.addWidget(self.grid_area)

    def open_webcam(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        popup = WebcamPopup(self.model, self.device, parent=self)
        popup.exec_()


    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Checkpoint (*.pt)")
        if path:
            try:
                ckpt = torch.load(path, map_location=self.device)
                meta = ckpt.get("metadata", {})
                state = ckpt.get("model_state", ckpt)

                choice = meta.get("model_choice", "").lower()
                ModelClass = model_registry.get(choice)
                if ModelClass is None:
                    raise ValueError("Unrecognized model type.")

                model = ModelClass()
                model.load_state_dict(state)
                model.to(self.device)
                model.eval()
                self.model = model

                QMessageBox.information(self, "Success", "Model loaded successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load model:\n{e}")

    def select_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset CSV", "", "CSV Files (*.csv)")
        if path:
            self.dataset = SignLanguageDataset.from_csv(path, transform=None)
            self._populate_grid()

    def _populate_grid(self):
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for i, (img, lbl) in enumerate(self.dataset):
            img_np = img.numpy().squeeze() if isinstance(img, torch.Tensor) else img
            qimg = QImage((img_np * 255).astype(np.uint8), 28, 28, 28, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(64, 64, Qt.KeepAspectRatio)

            thumb = QLabel()
            thumb.setPixmap(pix)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setToolTip(f"Label: {lbl}")
            thumb.mousePressEvent = lambda e, idx=i: self._open_popup(idx)

            self.grid_layout.addWidget(thumb, i // 6, i % 6)

    def _open_popup(self, index):
        if not self.model or not self.dataset:
            QMessageBox.warning(self, "Missing", "Please load a model and dataset first.")
            return

        img, _ = self.dataset[index]
        img_np = img.numpy().squeeze() if isinstance(img, torch.Tensor) else img
        popup = PredictionPopup(self.model, self.device, img_np, parent=self)
        popup.exec_()
