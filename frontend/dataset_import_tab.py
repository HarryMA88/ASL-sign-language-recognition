import os
import shutil
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QLabel,
    QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from backend.import_thread import ImportThread

class DatasetImportTab(QWidget):
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()

    def __init__(self):
        super().__init__()
        # Layouts
        self.layout = QVBoxLayout(self)
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

        self.layout.addLayout(btn_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Status label
        self.lbl_status = QLabel("No dataset loaded")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.lbl_status)

        # Success image (hidden until completion)
        self.success_label = QLabel()
        self.success_label.setAlignment(Qt.AlignCenter)
        self.success_label.setVisible(False)
        self.layout.addWidget(self.success_label)

        # Stop Import button
        self.btn_stop = QPushButton("Stop Import")
        self.btn_stop.setMinimumSize(100, 30)
        self.btn_stop.clicked.connect(self.stop_import)
        self.layout.addWidget(self.btn_stop)

        # Setup
        self.import_thread = None
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)
        self.update_buttons_state()

        # Auto-load existing CSV
        existing = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if existing:
            self._start_import_thread(os.path.join(self.data_folder, existing[0]))

    def update_buttons_state(self):
        csvs = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        has_data = bool(csvs)
        self.btn_import.setVisible(not has_data)
        self.btn_remove.setVisible(has_data)
        running = getattr(self.import_thread, '_is_running', False)
        self.btn_stop.setVisible(running)
        # Reset status when no data
        if not has_data:
            self.lbl_status.setText("No dataset loaded")
            self.lbl_status.setStyleSheet("")

    def import_dataset(self):
        if any(f.lower().endswith('.csv') for f in os.listdir(self.data_folder)):
            QMessageBox.warning(self, "Warning", "Remove existing dataset first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        dest = os.path.join(self.data_folder, os.path.basename(path))
        shutil.copy(path, dest)
        self._start_import_thread(dest)

    def _start_import_thread(self, path):
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Loading dataset...")
        self.lbl_status.setStyleSheet("")
        self.success_label.clear()
        self.success_label.setVisible(False)
        self.import_thread = ImportThread(path)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.on_import_finished)
        self.import_thread.start()
        self.update_buttons_state()

    @pyqtSlot(int, str)
    def update_progress(self, val, eta):
        self.progress_bar.setValue(val)
        if eta.startswith('-'):
            eta = "0m 0s"
        self.lbl_status.setText(f"ETA: {eta}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_import_finished(self, images, labels, shape):
        self.progress_bar.setValue(100)
        # Update status text
        self.lbl_status.setText("UPLOAD COMPLETE")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 16pt;")
        # Show success image
        pix = QPixmap('uploadComplete.png')
        if not pix.isNull():
            pix = pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.success_label.setPixmap(pix)
            self.success_label.setVisible(True)
        self.dataset_loaded.emit(images, labels, shape)
        self.update_buttons_state()

    def stop_import(self):
        if self.import_thread and getattr(self.import_thread, '_is_running', False):
            self.import_thread.stop()
            self.lbl_status.setText("Import stopped")

    def remove_dataset(self):
        if self.import_thread and getattr(self.import_thread, '_is_running', False):
            self.import_thread.stop()
        for f in os.listdir(self.data_folder):
            if f.lower().endswith('.csv'):
                os.remove(os.path.join(self.data_folder, f))
        # Clear UI
        self.progress_bar.setValue(0)
        self.lbl_status.setText("No dataset loaded")
        self.lbl_status.setStyleSheet("")
        self.success_label.clear()
        self.success_label.setVisible(False)
        self.dataset_cleared.emit()
        self.update_buttons_state()
