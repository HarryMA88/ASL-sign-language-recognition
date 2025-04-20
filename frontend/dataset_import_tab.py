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
    Tab for importing and clearing CSV datasets.

    Signals:
      - dataset_loaded(images, labels, shape): emitted on successful import
      - dataset_cleared(): emitted when datasets are removed
    """
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Apply dark theme to all widgets
        self.setStyleSheet("""
        QWidget { background: #2E2E2E; color: #EEEEEE; }
        QPushButton { background: #444; color: #EEE; border: none; border-radius:5px; padding:8px; font-weight:bold; }
        QPushButton:hover { background: #555; }
        QProgressBar { background: #CCC; border:1px solid #555; border-radius:5px; text-align:center; }
        QProgressBar::chunk { background: #555; }
        QLabel { color: #CCC; }
        """
        )

        # Main layout and button row
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()

        # "Add Dataset" button
        self.btn_import = QPushButton("Add Dataset")
        self.btn_import.setMinimumSize(150, 40)
        self.btn_import.clicked.connect(self.import_dataset)
        btn_layout.addWidget(self.btn_import)

        # "Remove Dataset" button
        self.btn_remove = QPushButton("Remove Dataset")
        self.btn_remove.setMinimumSize(150, 40)
        self.btn_remove.clicked.connect(self.remove_dataset)
        btn_layout.addWidget(self.btn_remove)

        layout.addLayout(btn_layout)

        # Progress bar for import thread
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Button to stop the import process
        self.btn_stop_loading = QPushButton("Stop Loading")
        self.btn_stop_loading.setMinimumSize(150, 30)
        self.btn_stop_loading.clicked.connect(self.stop_loading)
        self.btn_stop_loading.hide()  # hidden until import starts
        layout.addWidget(self.btn_stop_loading)

        # Status label for import progress or messages
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        # Success icon, displayed on completion
        self.success_label = QLabel()
        self.success_label.setAlignment(Qt.AlignCenter)
        self.success_label.setVisible(False)
        layout.addWidget(self.success_label)

        # Placeholder text and icon when no dataset is loaded
        self.placeholder_label = QLabel("No Data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-weight:bold; font-size:16pt;")
        layout.addWidget(self.placeholder_label)

        self.placeholder_icon = QLabel()
        self.placeholder_icon.setAlignment(Qt.AlignCenter)
        pix = QPixmap('redCross.png')
        if not pix.isNull():
            pix = pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.placeholder_icon.setPixmap(pix)
        layout.addWidget(self.placeholder_icon)

        # Prepare import thread and data directory
        self.import_thread = None
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)

        # Hide "Remove" until a dataset is present
        self.btn_remove.hide()

        # Auto-load existing CSV if found
        existing = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if existing:
            csv_path = os.path.join(self.data_folder, existing[0])
            self._start_import_thread(csv_path)
            self.btn_remove.show()
            self.btn_import.hide()
            self.placeholder_label.hide()
            self.placeholder_icon.hide()

    def import_dataset(self):
        """Open file dialog, copy selected CSV, and start import."""
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return  # user cancelled

        # Copy chosen file into local data folder
        dest = os.path.join(self.data_folder, os.path.basename(path))
        shutil.copy(path, dest)

        # Begin import and adjust UI
        self._start_import_thread(dest)
        self.btn_import.hide()
        self.btn_remove.show()
        self.placeholder_label.hide()
        self.placeholder_icon.hide()

    def _start_import_thread(self, path):
        """Initialize and run the background thread to import the CSV."""
        # Reset UI elements for new import
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Loading dataset...")
        self.success_label.setVisible(False)
        self.btn_stop_loading.show()

        # Launch thread and connect signals
        self.import_thread = ImportThread(path)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.on_import_finished)
        self.import_thread.start()

    @pyqtSlot(int, str)
    def update_progress(self, val, eta):
        """Update progress bar and status text."""
        self.progress_bar.setValue(val)
        # Show ETA, default to 0 if negative
        status = eta if not eta.startswith('-') else '0m 0s'
        self.lbl_status.setText(f"ETA: {status}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_import_finished(self, images, labels, shape):
        """Finalize UI and emit loaded signal when import completes."""
        self.progress_bar.setValue(100)
        self.lbl_status.setText("UPLOAD COMPLETE")
        self.lbl_status.setStyleSheet("font-weight:bold; font-size:16pt;")
        self.btn_stop_loading.hide()

        # Display completion icon
        pix = QPixmap('uploadComplete.png')
        if not pix.isNull():
            pix = pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.success_label.setPixmap(pix)
            self.success_label.setVisible(True)

        # Notify listeners that dataset is ready
        self.dataset_loaded.emit(images, labels, shape)

    def stop_loading(self):
        """Stop the import thread and update status."""
        if self.import_thread:
            self.import_thread.stop()
        self.lbl_status.setText("LOADING HALTED")
        self.btn_stop_loading.hide()

    def remove_dataset(self):
        """Delete any CSV files and reset the UI to initial state."""
        # Remove all CSVs in data folder
        for fname in os.listdir(self.data_folder):
            if fname.lower().endswith('.csv'):
                try:
                    os.remove(os.path.join(self.data_folder, fname))
                except OSError:
                    pass

        # Reset UI elements
        self.progress_bar.setValue(0)
        self.lbl_status.clear()
        self.success_label.clear()
        self.success_label.setVisible(False)
        self.btn_stop_loading.hide()
        self.btn_import.show()
        self.btn_remove.hide()
        self.placeholder_label.show()
        self.placeholder_icon.show()
        self.dataset_cleared.emit()
