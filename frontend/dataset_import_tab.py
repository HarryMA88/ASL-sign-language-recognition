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
    A tab where you can load CSV datasets. When loading finishes, it’ll let the rest of the app know,
    and you can clear out datasets too. If there’s already a CSV in the data folder, it loads automatically.
    """
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Grey Background
        self.setStyleSheet("""
        QWidget { background: #2E2E2E; color: #EEEEEE; }
        QPushButton { background: #444; color: #EEE; border: none; border-radius:5px; padding:8px; font-weight:bold; }
        QPushButton:hover { background: #555; }
        QProgressBar { background: #CCC; border:1px solid #555; border-radius:5px; text-align:center; }
        QProgressBar::chunk { background: #555; }
        QLabel { color: #CCC; }
        """)

        # Set up layouts for buttons and widgets
        layout = QVBoxLayout(self)
        btn_layout = QHBoxLayout()

        # Button to add a new dataset
        self.btn_import = QPushButton("Add Dataset")
        self.btn_import.setMinimumSize(150, 40)
        self.btn_import.clicked.connect(self.import_dataset)
        btn_layout.addWidget(self.btn_import)

        # Button to remove the current dataset
        self.btn_remove = QPushButton("Remove Dataset")
        self.btn_remove.setMinimumSize(150, 40)
        self.btn_remove.clicked.connect(self.remove_dataset)
        btn_layout.addWidget(self.btn_remove)

        layout.addLayout(btn_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Stop iport button 
        self.btn_stop_loading = QPushButton("Stop Loading")
        self.btn_stop_loading.setMinimumSize(150, 30)
        self.btn_stop_loading.clicked.connect(self.stop_loading)
        self.btn_stop_loading.hide()
        layout.addWidget(self.btn_stop_loading)

        #Status label
        self.lbl_status = QLabel("")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)


        self.success_label = QLabel()
        self.success_label.setAlignment(Qt.AlignCenter)
        self.success_label.setVisible(False)
        layout.addWidget(self.success_label)

        # Text to show when nothing loaded
        self.placeholder_label = QLabel("No Data")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("font-weight:bold; font-size:16pt;")
        layout.addWidget(self.placeholder_label)

        # Icon to show there’s no data 
        self.placeholder_icon = QLabel()
        self.placeholder_icon.setAlignment(Qt.AlignCenter)
        placeholder_pix = QPixmap('redCross.png')
        if not placeholder_pix.isNull():
            placeholder_pix = placeholder_pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.placeholder_icon.setPixmap(placeholder_pix)
        layout.addWidget(self.placeholder_icon)

        # Prepare the import thread variable and ensure the data folder exists
        self.import_thread = None
        self.data_folder = "data"
        os.makedirs(self.data_folder, exist_ok=True)

        self.btn_remove.hide()

        # Check if there’s already a CSV waiting and load it right away
        existing = [f for f in os.listdir(self.data_folder) if f.lower().endswith('.csv')]
        if existing:
            csv_path = os.path.join(self.data_folder, existing[0])
            self._start_import_thread(csv_path)
            self.btn_remove.show()
            self.btn_import.hide()
            self.placeholder_label.hide()
            self.placeholder_icon.hide()

    def import_dataset(self):
        # Let the user pick a CSV and copy it into our data folder
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return

        dest = os.path.join(self.data_folder, os.path.basename(path))
        shutil.copy(path, dest)

        # Kick off the import and update the buttons and placeholders
        self._start_import_thread(dest)
        self.btn_import.hide()
        self.btn_remove.show()
        self.placeholder_label.hide()
        self.placeholder_icon.hide()

    def _start_import_thread(self, path):
        # Get the UI ready for import
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Loading dataset...")
        self.success_label.setVisible(False)
        self.btn_stop_loading.show()

        # Launch the background thread
        self.import_thread = ImportThread(path)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.on_import_finished)
        self.import_thread.start()

    @pyqtSlot(int, str)
    def update_progress(self, val, eta):
        # Update the progress bar and ETA message
        self.progress_bar.setValue(val)
        self.lbl_status.setText(f"ETA: {eta if not eta.startswith('-') else '0m 0s'}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_import_finished(self, images, labels, shape):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("UPLOAD COMPLETE")
        self.lbl_status.setStyleSheet("font-weight:bold; font-size:16pt;")
        self.btn_stop_loading.hide()

        pix = QPixmap('uploadComplete.png')
        if not pix.isNull():
            pix = pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.success_label.setPixmap(pix)
            self.success_label.setVisible(True)

        # Shares data across app
        self.dataset_loaded.emit(images, labels, shape)

    def stop_loading(self):
        # Stop the thread if it’s still going and let the user know it stopped
        if self.import_thread:
            self.import_thread.stop()
        self.lbl_status.setText("LOADING HALTED")
        self.btn_stop_loading.hide()

    def remove_dataset(self):
        # Clear out any CSV
        for fname in os.listdir(self.data_folder):
            if fname.lower().endswith('.csv'):
                try:
                    os.remove(os.path.join(self.data_folder, fname))
                except OSError:
                    pass

        #Reverts back to originol
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
