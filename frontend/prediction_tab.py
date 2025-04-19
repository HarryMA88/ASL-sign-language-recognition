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
    """
    Tab for running predictions either from a webcam stream or a saved dataset.

    - Open webcam feed
    - Load a pretrained model
    - Load a CSV dataset and browse thumbnails
    - Click a thumbnail to see detailed prediction
    """
    def __init__(self):
        super().__init__()
        # Keep track of model, dataset and thumbnail state
        self.model = None
        self.dataset = None
        self.all_data = []
        self.total = 0
        self.page_size = 0
        self.max_idx = 0
        # Auto-detect GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Main layout and top control bar
        self.layout = QVBoxLayout(self)
        self.control_bar = QHBoxLayout()
        self.layout.addLayout(self.control_bar)

        # Buttons for webcam, model, and dataset
        self.btn_webcam = QPushButton("Open Webcam")
        self.btn_model = QPushButton("Select Model")
        self.btn_data = QPushButton("Select Dataset")

        # Wire up button clicks
        self.btn_webcam.clicked.connect(self.open_webcam)
        self.btn_model.clicked.connect(self.select_model)
        self.btn_data.clicked.connect(self.select_dataset)

        # Add buttons to the control bar
        for btn in (self.btn_webcam, self.btn_model, self.btn_data):
            self.control_bar.addWidget(btn)

        # Scroll area for thumbnail grid
        self.grid_area = QScrollArea()
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_area.setWidget(self.grid_widget)
        self.grid_area.setWidgetResizable(True)
        self.layout.addWidget(self.grid_area)

        # Trigger infinite scroll when reaching bottom
        self.grid_area.verticalScrollBar().valueChanged.connect(self._on_scroll)

    def open_webcam(self):
        """Show live webcam predictions if a model is loaded."""
        if not self.model:
            QMessageBox.warning(self, "No Model", "Load a model first, then try webcam.")
            return
        popup = WebcamPopup(self.model, self.device, parent=self)
        popup.exec_()

    def select_model(self):
        """Pick a .pt checkpoint, load it, and prepare for inference."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "PyTorch Checkpoint (*.pt)")
        if not path:
            return
        try:
            ckpt = torch.load(path, map_location=self.device)
            meta = ckpt.get("metadata", {})
            state = ckpt.get("model_state", ckpt)
            choice = meta.get("model_choice", "").lower()
            ModelClass = model_registry.get(choice)
            if not ModelClass:
                raise ValueError("Unrecognised model type")
            model = ModelClass()
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            self.model = model
            QMessageBox.information(self, "Success", "Model loaded!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    def select_dataset(self):
        """Load a CSV, clear old thumbnails, and start background loading."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        self._clear_thumbnails()
        self.loader = DatasetLoader(path)
        self.loader.datasetLoaded.connect(self._set_dataset)
        self.loader.finished.connect(self._on_load_finished)
        self.loader.start()

    def _clear_thumbnails(self):
        """Helper to remove all widgets from the thumbnail grid."""
        for i in reversed(range(self.grid_layout.count())):
            w = self.grid_layout.itemAt(i).widget()
            if w:
                w.setParent(None)

    def _set_dataset(self, ds):
        """Receive the loaded dataset object."""
        self.dataset = ds

    def _on_load_finished(self):
        """Once CSV loading wraps up, build and show thumbnails."""
        # Turn the 'Select Dataset' button into a 'Clear Dataset' button
        self.btn_data.setText("Clear Dataset")
        try:
            self.btn_data.clicked.disconnect()
        except TypeError:
            pass
        self.btn_data.clicked.connect(self.clear_dataset)

        # Prepare thumbnail pixmaps and labels
        thumb_size = QSize(100, 100)
        self.all_data.clear()
        for img, lbl in self.dataset:
            arr = img.numpy().squeeze() if torch.is_tensor(img) else img
            h, w = arr.shape
            qimg = QImage((arr * 255).astype('uint8').data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.all_data.append((pix, lbl))

        # Set up paging info and load the first batch
        self.total = len(self.all_data)
        self.page_size = math.ceil(self.total / 10) if self.total else 0
        self.max_idx = 0
        self._add_next_page()

    def clear_dataset(self):
        """Clear loaded data and reset the UI to initial state."""
        self.dataset = None
        self.all_data.clear()
        self.total = self.page_size = self.max_idx = 0
        self._clear_thumbnails()

        # Swap button back to loading mode
        self.btn_data.setText("Select Dataset")
        try:
            self.btn_data.clicked.disconnect()
        except TypeError:
            pass
        self.btn_data.clicked.connect(self.select_dataset)

    def _add_next_page(self):
        """Append the next batch of thumbnails to the grid."""
        start = self.max_idx
        end = min(self.max_idx + self.page_size, self.total)
        for idx in range(start, end):
            pix, lbl = self.all_data[idx]
            thumb = QLabel()
            thumb.setPixmap(pix)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setToolTip(f"Label: {lbl}")
            thumb.mousePressEvent = lambda e, i=idx: self._open_popup(i)
            self.grid_layout.addWidget(thumb, idx // 6, idx % 6)
        self.max_idx = end

    def _on_scroll(self, val):
        """Load more thumbnails when scrolling near the bottom."""
        sb = self.grid_area.verticalScrollBar()
        if val >= sb.maximum() - 10 and self.max_idx < self.total:
            self._add_next_page()

    def _open_popup(self, index):
        """Show a detailed prediction popup when a thumbnail is clicked."""
        if not self.model or not self.dataset:
            QMessageBox.warning(self, "Missing", "Load both a model and dataset first.")
            return
        img, _ = self.dataset[index]
        arr = img.numpy().squeeze() if torch.is_tensor(img) else img
        popup = PredictionPopup(self.model, self.device, arr, parent=self)
        popup.exec_()