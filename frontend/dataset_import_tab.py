import os
import shutil
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QLabel,
    QFileDialog
)
from PyQt5.QtGui import QPixmap
from backend.import_thread import ImportThread

class DatasetImportTab(QWidget):
    """
    Tab for importing CSV datasets. Emits dataset_loaded when import completes,
    and dataset_cleared when datasets are removed.
    Automatically loads any existing CSV on startup.
    """
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Apply dark theme stylesheet
        self.setStyleSheet("""
        QWidget { background: #2E2E2E; color: #EEEEEE; }
        QPushButton { background: #444; color: #EEE; border: none; border-radius:5px; padding:8px; font-weight:bold; }
        QPushButton:hover { background: #555; }
        QProgressBar { background: #CCC; border:1px solid #555; border-radius:5px; text-align:center; }
        QProgressBar::chunk { background: #555; }
        QLabel { color: #CCC; }
        """)

        # Layouts
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()

        # Add Dataset button
        self.btn_import = QPushButton("Add Dataset")
        self.btn_import.setMinimumSize(150, 40)
        self.btn_import.clicked.connect(self.import_dataset)
        btn_layout.addWidget(self.btn_import)

        # Remove Dataset button
        self.btn_remove = QPushButton("Remove Dataset")
        self.btn_remove.setMinimumSize(150, 40)
        self.btn_remove.clicked.connect(self.remove_dataset)
        btn_layout.addWidget(self.btn_remove)

        layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Stop Loading button (hidden until import starts)
        self.btn_stop_loading = QPushButton("Stop Loading")
        self.btn_stop_loading.setMinimumSize(150, 30)
        self.btn_stop_loading.clicked.connect(self.stop_loading)
        self.btn_stop_loading.hide()
        layout.addWidget(self.btn_stop_loading)

        # Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        # Success image
        self.success_label = QLabel()
        self.success_label.setAlignment(Qt.AlignCenter)
        self.success_label.setVisible(False)
        layout.addWidget(self.success_label)

        # Placeholder text for no data
        self.placeholder_label = QLabel("No Data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-weight:bold; font-size:16pt;")
        layout.addWidget(self.placeholder_label)

        # Placeholder icon
        self.placeholder_icon = QLabel()
        self.placeholder_icon.setAlignment(Qt.AlignCenter)
        pix = QPixmap('redCross.png')
        if not pix.isNull():
            pix = pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.placeholder_icon.setPixmap(pix)
        layout.addWidget(self.placeholder_icon)

        # Setup
        self.import_thread = None
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)

        # Auto-load existing CSV if present
        existing = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if existing:
            csv_path = os.path.join(self.data_folder, existing[0])
            self._start_import_thread(csv_path)
            self.btn_remove.setVisible(True)
            self.btn_import.setVisible(False)
            self.placeholder_label.setVisible(False)
            self.placeholder_icon.setVisible(False)
        else:
            self.btn_remove.setVisible(False)
            self.placeholder_label.setVisible(True)
            self.placeholder_icon.setVisible(True)

    def import_dataset(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        dest = os.path.join(self.data_folder, os.path.basename(path))
        shutil.copy(path, dest)
        self._start_import_thread(dest)
        self.btn_import.setVisible(False)
        self.btn_remove.setVisible(True)
        self.placeholder_label.setVisible(False)
        self.placeholder_icon.setVisible(False)

    def _start_import_thread(self, path):
        # Reset UI for new import
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Loading dataset...")
        self.success_label.setVisible(False)
        # Show stop button
        self.btn_stop_loading.show()

        # Start thread
        self.import_thread = ImportThread(path)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.on_import_finished)
        self.import_thread.start()

    @pyqtSlot(int, str)
    def update_progress(self, val, eta):
        self.progress_bar.setValue(val)
        self.lbl_status.setText(f"ETA: {eta if not eta.startswith('-') else '0m 0s'}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_import_finished(self, images, labels, shape):
        # Complete UI
        self.progress_bar.setValue(100)
        self.lbl_status.setText("UPLOAD COMPLETE")
        self.lbl_status.setStyleSheet("font-weight:bold; font-size:16pt;")
        # Hide stop button
        self.btn_stop_loading.hide()

        # Show success image
        pix = QPixmap('uploadComplete.png')
        if not pix.isNull():
            pix = pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.success_label.setPixmap(pix)
            self.success_label.setVisible(True)

        self.dataset_loaded.emit(images, labels, shape)

    def stop_loading(self):
        if self.import_thread:
            self.import_thread.stop()         # stop the background import
        self.lbl_status.setText("LOADING HALTED")
        self.btn_stop_loading.hide()

    def remove_dataset(self):
        # Safely remove CSV files
        for f in os.listdir(self.data_folder):
            if f.lower().endswith('.csv'):
                try:
                    os.remove(os.path.join(self.data_folder, f))
                except OSError:
                    pass

        # Reset UI
        self.progress_bar.setValue(0)
        self.lbl_status.clear()
        self.success_label.clear()
        self.success_label.setVisible(False)
        self.btn_stop_loading.hide()
        self.btn_import.setVisible(True)
        self.btn_remove.setVisible(False)
        self.placeholder_label.setVisible(True)
        self.placeholder_icon.setVisible(True)
        self.dataset_cleared.emit()
