import torch
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox,
    QScrollArea, QGridLayout
)
from ml.models import model_registry
from backend.dataset_loader import DatasetLoader
from frontend.prediction_popup import PredictionPopup
from frontend.webcam_popup import WebcamPopup

class PredictionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model   = None
        self.dataset = None
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # layout setup
        self.layout      = QVBoxLayout(self)
        self.control_bar = QHBoxLayout()
        self.grid_area   = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)

        # buttons
        self.btn_webcam = QPushButton("Open Webcam")
        self.btn_model  = QPushButton("Select Model")
        self.btn_data   = QPushButton("Select Dataset")

        self.btn_webcam.clicked.connect(self.open_webcam)
        self.btn_model.clicked.connect(self.select_model)
        self.btn_data.clicked.connect(self.select_dataset)

        self.control_bar.addWidget(self.btn_webcam)
        self.control_bar.addWidget(self.btn_model)
        self.control_bar.addWidget(self.btn_data)
        self.layout.addLayout(self.control_bar)

        # scroll area for thumbnails
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
        if not path:
            return

        try:
            ckpt  = torch.load(path, map_location=self.device)
            meta  = ckpt.get("metadata", {})
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
        if not path:
            return

        # clear any existing thumbnails
        for i in reversed(range(self.grid_layout.count())):
            widget = self.grid_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        # start loader thread
        self.loader = DatasetLoader(path)
        self.loader.datasetLoaded.connect(self._set_dataset)
        self.loader.batchReady.connect(self._add_thumbnails)
        self.loader.finished.connect(self._on_load_finished)
        self.loader.start()

    def _set_dataset(self, ds):
        self.dataset = ds

    def _add_thumbnails(self, batch):
        base = self.grid_layout.count()
        for pix, lbl in batch:
            thumb = QLabel()
            thumb.setPixmap(pix)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setToolTip(f"Label: {lbl}")

            idx = base
            thumb.mousePressEvent = lambda e, i=idx: self._open_popup(i)
            self.grid_layout.addWidget(thumb, idx // 6, idx % 6)
            base += 1

    def _on_load_finished(self):
        # disable and relabel the dataset button
        self.btn_data.setText("Dataset Loaded")
        self.btn_data.setEnabled(False)

    def _open_popup(self, index):
        if not self.model or not self.dataset:
            QMessageBox.warning(self, "Missing", "Please load a model and dataset first.")
            return

        img, _ = self.dataset[index]
        img_np  = img.numpy().squeeze() if isinstance(img, torch.Tensor) else img
        popup   = PredictionPopup(self.model, self.device, img_np, parent=self)
        popup.exec_()
