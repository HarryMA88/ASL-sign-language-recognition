import math
import torch
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QScrollArea, QGridLayout
)
from ml.models import model_registry
from backend.dataset_loader import DatasetLoader
from frontend.prediction_popup import PredictionPopup
from frontend.webcam_popup import WebcamPopup

class PredictionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model     = None
        self.dataset   = None
        self.all_data  = []
        self.total     = 0
        self.page_size = 0
        self.max_idx   = 0
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layout      = QVBoxLayout(self)
        self.control_bar = QHBoxLayout()
        self.layout.addLayout(self.control_bar)

        self.btn_webcam = QPushButton("Open Webcam")
        self.btn_model  = QPushButton("Select Model")
        self.btn_data   = QPushButton("Select Dataset")

        self.btn_webcam.clicked.connect(self.open_webcam)
        self.btn_model.clicked.connect(self.select_model)
        self.btn_data.clicked.connect(self.select_dataset)

        self.control_bar.addWidget(self.btn_webcam)
        self.control_bar.addWidget(self.btn_model)
        self.control_bar.addWidget(self.btn_data)

        self.grid_area   = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)

        self.grid_area.setWidget(self.grid_widget)
        self.grid_area.setWidgetResizable(True)
        self.layout.addWidget(self.grid_area)

        self.grid_area.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def open_webcam(self):
        if not self.model:
            QMessageBox.warning(self, "No Model", "Please load a model first.")
            return
        popup = WebcamPopup(self.model, self.device, parent=self)
        popup.exec_()

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Checkpoint (*.pt)")
        if not path:
            return

        try:
            ckpt  = torch.load(path, map_location=self.device)
            meta  = ckpt.get("metadata", {})
            state = ckpt.get("model_state", ckpt)

            choice      = meta.get("model_choice", "").lower()
            ModelClass  = model_registry.get(choice)
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
        if not path:
            return

        # clear out any existing thumbnails
        for i in reversed(range(self.grid_layout.count())):
            w = self.grid_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        # load new dataset in background
        self.loader = DatasetLoader(path)
        self.loader.datasetLoaded.connect(self._set_dataset)
        self.loader.finished.connect(self._on_load_finished)
        self.loader.start()

    def _set_dataset(self, ds):
        self.dataset = ds

    def _on_load_finished(self):
        # repurpose the button into a “Clear Dataset” action
        self.btn_data.setText("Clear Dataset")
        try:
            self.btn_data.clicked.disconnect()
        except TypeError:
            pass
        self.btn_data.clicked.connect(self.clear_dataset)

        # build thumbnails
        thumb_size = QSize(100, 100)
        self.all_data.clear()
        for img, lbl in self.dataset:
            arr = img.numpy().squeeze() if torch.is_tensor(img) else img
            h, w = arr.shape
            qimg = QImage((arr * 255).astype('uint8').data, w, h, w, QImage.Format_Grayscale8)
            pix  = QPixmap.fromImage(qimg).scaled(thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.all_data.append((pix, lbl))

        self.total     = len(self.all_data)
        self.page_size = math.ceil(self.total / 10) if self.total else 0
        self.max_idx   = 0
        self._add_next_page()

    def clear_dataset(self):
        # reset state
        self.dataset   = None
        self.all_data.clear()
        self.total     = 0
        self.page_size = 0
        self.max_idx   = 0

        # remove thumbnails
        for i in reversed(range(self.grid_layout.count())):
            w = self.grid_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

        # restore button to loading mode
        self.btn_data.setText("Select Dataset")
        try:
            self.btn_data.clicked.disconnect()
        except TypeError:
            pass
        self.btn_data.clicked.connect(self.select_dataset)

    def _add_next_page(self):
        base = self.max_idx
        end  = min(self.max_idx + self.page_size, self.total)
        for idx in range(base, end):
            pix, lbl = self.all_data[idx]
            thumb = QLabel()
            thumb.setPixmap(pix)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setToolTip(f"Label: {lbl}")
            thumb.mousePressEvent = lambda e, i=idx: self._open_popup(i)
            self.grid_layout.addWidget(thumb, idx // 6, idx % 6)
        self.max_idx = end

    def _on_scroll(self, val):
        sb = self.grid_area.verticalScrollBar()
        if val >= sb.maximum() - 10 and self.max_idx < self.total:
            self._add_next_page()

    def _open_popup(self, index):
        if not self.model or not self.dataset:
            QMessageBox.warning(self, "Missing", "Please load a model and dataset first.")
            return

        img, _ = self.dataset[index]
        arr    = img.numpy().squeeze() if torch.is_tensor(img) else img
        popup  = PredictionPopup(self.model, self.device, arr, parent=self)
        popup.exec_()
