import os
import shutil
import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QProgressBar,
    QLabel, QFileDialog, QMessageBox
)
from backend.import_thread import ImportThread

class DatasetImportTab(QWidget):
    dataset_loaded  = pyqtSignal(np.ndarray, np.ndarray, tuple)
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

<<<<<<< Updated upstream
        # Prepare data folder
        # (assumes 'data' lives alongside frontend/)
        self.data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
=======
        self.import_thread = None
        self.data_folder = "data"
>>>>>>> Stashed changes
        os.makedirs(self.data_folder, exist_ok=True)

<<<<<<< Updated upstream
        # Initial button state: no CSV imported yet
        self.btn_import.setEnabled(True)
        self.btn_remove.setEnabled(False)

        self.import_thread = None

    def update_buttons_state(self):
        csvs = [f for f in os.listdir(self.data_folder) if f.lower().endswith(".csv")]
        self.btn_import.setEnabled(len(csvs) == 0)
        self.btn_remove.setEnabled(len(csvs) > 0)

    def import_dataset(self):
        # Prevent multiple CSVs
        csvs = [f for f in os.listdir(self.data_folder) if f.lower().endswith(".csv")]
        if csvs:
            QMessageBox.warning(self, "Warning", "Please remove the existing CSV before importing a new one.")
=======
        # Auto-load existing CSV if present
        existing = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if existing:
            path = os.path.join(self.data_folder, existing[0])
            self.btn_import.setEnabled(False)
            self.btn_remove.setEnabled(True)
            self._start_import_thread(path)

    def update_buttons_state(self):
        csvs = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        has_data = bool(csvs)
        self.btn_import.setVisible(not has_data)
        self.btn_remove.setVisible(has_data)
        # stop button only if import thread is active
        running = getattr(self.import_thread, '_is_running', False)
        self.btn_stop.setVisible(running)

    def import_dataset(self):
        # Only allow one CSV
        existing = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if existing:
            QMessageBox.warning(self, "Warning", "Remove existing dataset first.")
>>>>>>> Stashed changes
            return

        path, _ = QFileDialog.getOpenFileName(self, "Select CSV File", "", "CSV Files (*.csv)")
        if not path:
            return

        dest = os.path.join(self.data_folder, os.path.basename(path))
<<<<<<< Updated upstream
        try:
            shutil.copy(path, dest)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy file:\n{e}")
            return

        # Disable import, enable stop
        self.update_buttons_state()
        self.btn_stop.setEnabled(True)

        # Start background import
        self.import_thread = ImportThread(dest)
=======
        shutil.copy(path, dest)
        self.btn_import.setEnabled(False)
        self.btn_remove.setEnabled(True)
        self._start_import_thread(dest)

    def _start_import_thread(self, path):
        self.progress_bar.setValue(0)
        self.lbl_eta.setText("Loading dataset...")
        self.btn_stop.setEnabled(True)
        self.import_thread = ImportThread(path)
>>>>>>> Stashed changes
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.on_import_finished)
        self.import_thread.start()

    @pyqtSlot(int, str)
    def update_progress(self, value, eta):
        self.progress_bar.setValue(value)
        self.lbl_eta.setText(f"ETA: {eta}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_import_finished(self, images, labels, shape):
        # Stop button no longer needed once finished
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_eta.setText("Import complete")

        # Notify listeners
        self.dataset_loaded.emit(images, labels, shape)

        # Update import/remove button states
        self.update_buttons_state()

    def stop_import(self):
        if self.import_thread:
            self.import_thread.stop()
            self.btn_stop.setEnabled(False)
            self.lbl_eta.setText("Import stopped")

    def remove_dataset(self):
        # Delete any CSV in data_folder
        for f in os.listdir(self.data_folder):
            if f.lower().endswith(".csv"):
                try:
                    os.remove(os.path.join(self.data_folder, f))
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to remove {f}:\n{e}")

        # Reset UI
        self.progress_bar.setValue(0)
        self.lbl_eta.setText("No dataset loaded")
        self.dataset_cleared.emit()
        self.update_buttons_state()
