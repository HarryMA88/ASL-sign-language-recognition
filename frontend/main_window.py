import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import pyqtSlot
from frontend.dataset_import_tab import DatasetImportTab
from frontend.dataset_viewer_tab import DatasetViewerTab
from frontend.training_tab import TrainingTab
from frontend.prediction_tab import PredictionTab
from backend.dataset import SignLanguageDataset

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Sign Language Recognition Tool")
        self.resize(1000, 800)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.import_tab = DatasetImportTab()
        self.viewer_tab = DatasetViewerTab()
        self.training_tab = TrainingTab()
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.import_tab, "Dataset Import")
        self.tabs.addTab(self.viewer_tab, "Dataset Viewer")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.prediction_tab, "Prediction")
        self.import_tab.dataset_loaded.connect(self.on_dataset_loaded)
        self.import_tab.dataset_cleared.connect(self.on_dataset_cleared)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_dataset_loaded(self, images, labels, img_shape):
        self.dataset = SignLanguageDataset(images, labels)
        self.viewer_tab.load_dataset(images, labels)
        self.training_tab.set_dataset(self.dataset)
        self.training_tab.input_shape = img_shape

    @pyqtSlot()
    def on_dataset_cleared(self):
        self.dataset = None
        self.viewer_tab.load_dataset(np.array([]), np.array([]))
        self.training_tab.clear_dataset()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
