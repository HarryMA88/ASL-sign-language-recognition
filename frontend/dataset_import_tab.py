import os
import shutil
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QProgressBar, QLabel, QFileDialog, QMessageBox
from backend.import_thread import ImportThread

class DatasetImportTab(QWidget):
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        # Import button
        self.btn_import = QPushButton("Import Dataset CSV")
        self.btn_import.clicked.connect(self.import_dataset)
        self.layout.addWidget(self.btn_import)
        # Remove button
        self.btn_remove = QPushButton("Remove Imported Dataset")
        self.btn_remove.clicked.connect(self.remove_dataset)
        self.layout.addWidget(self.btn_remove)
        # Progress bar and ETA label
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.lbl_eta = QLabel("ETA: N/A")
        self.layout.addWidget(self.lbl_eta)
        # Stop button
        self.btn_stop = QPushButton("Stop Import")
        self.btn_stop.clicked.connect(self.stop_import)
        self.btn_stop.setEnabled(False)
        self.layout.addWidget(self.btn_stop)

        self.import_thread = None
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)
        self.update_buttons_state()

    def update_buttons_state(self):
        csvs = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        self.btn_import.setEnabled(len(csvs) == 0)
        self.btn_remove.setEnabled(len(csvs) > 0)

    def import_dataset(self):
        csvs = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if csvs:
            QMessageBox.warning(self, "Warning", "Remove existing CSV first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        dest = os.path.join(self.data_folder, os.path.basename(path))
        shutil.copy(path, dest)
        self.btn_import.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.import_thread = ImportThread(dest)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.on_import_finished)
        self.import_thread.start()
        self.update_buttons_state()

    @pyqtSlot(int, str)
    def update_progress(self, val, eta):
        self.progress_bar.setValue(val)
        self.lbl_eta.setText(f"ETA: {eta}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_import_finished(self, images, labels, shape):
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_eta.setText("Import complete")
        self.dataset_loaded.emit(images, labels, shape)
        self.update_buttons_state()

    def stop_import(self):
        if self.import_thread:
            self.import_thread.stop()
            self.btn_stop.setEnabled(False)
            self.lbl_eta.setText("Import stopped")

    def remove_dataset(self):
        for f in os.listdir(self.data_folder):
            if f.lower().endswith('.csv'):
                os.remove(os.path.join(self.data_folder, f))
        self.progress_bar.setValue(0)
        self.lbl_eta.setText("No dataset loaded")
        self.dataset_cleared.emit()
        self.update_buttons_state()