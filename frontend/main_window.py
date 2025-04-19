import numpy as np
from PyQt5.QtWidgets import QMainWindow, QTabWidget
from PyQt5.QtCore import pyqtSlot
from frontend.dataset_import_tab import DatasetImportTab
from frontend.dataset_viewer_tab import DatasetViewerTab
from frontend.training_tab import TrainingTab
from frontend.prediction_tab import PredictionTab
from backend.dataset import SignLanguageDataset

class MainWindow(QMainWindow):
    """
    Main window for the Sign Language Recognition Tool.
    Contains tabs for importing, viewing, training and predicting datasets.
    """
    def __init__(self):
        super().__init__()

        # Window title and size
        self.setWindowTitle("Sign Language Recognition Tool")
        self.resize(1280, 900)

        # Apply global dark theme and Magistral Light font
        self.setStyleSheet("""
        QMainWindow { background-color: #2E2E2E; color: #EEEEEE;
                      font-family: 'Magistral Light', Arial, sans-serif; }
        QTabWidget::pane { background-color: #2E2E2E; border: none; }
        QTabBar::tab { background-color: #333333; color: #EEEEEE;
                       padding: 10px 40px; min-width: 240px; border: none;
                       border-top-left-radius: 4px; border-top-right-radius: 4px;
                       font-size: 14pt; }
        QTabBar::tab:selected { background-color: #555555; }
        QTabBar::tab:hover    { background-color: #444444; }
        """)

        # Set up tabs without auto-expanding, so each sizes to content
        self.tabs = QTabWidget(self)
        self.tabs.tabBar().setExpanding(False)
        self.setCentralWidget(self.tabs)

        # Instantiate each tab
        self.import_tab     = DatasetImportTab()
        self.viewer_tab     = DatasetViewerTab()
        self.training_tab   = TrainingTab()
        self.prediction_tab = PredictionTab()

        # Add tabs in order
        self.tabs.addTab(self.import_tab,     "Dataset Import")
        self.tabs.addTab(self.viewer_tab,     "Dataset Viewer")
        self.tabs.addTab(self.training_tab,   "Training")
        self.tabs.addTab(self.prediction_tab, "Prediction")

        # Wire up import signals to viewer and trainer
        self.import_tab.dataset_loaded.connect(self.on_dataset_loaded)
        self.import_tab.dataset_cleared.connect(self.on_dataset_cleared)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_dataset_loaded(self, images, labels, img_shape):
        """
        When new data arrives:
        - Wrap arrays in our dataset class
        - Push raw images to the viewer
        - Give the dataset object to the trainer
        """
        self.dataset = SignLanguageDataset(images, labels)
        self.viewer_tab.dataset_loaded(images, labels, img_shape)
        self.training_tab.set_dataset(self.dataset)

    @pyqtSlot()
    def on_dataset_cleared(self):
        """
        When the dataset is removed:
        - Clear the viewer display
        - Reset the trainer
        """
        self.viewer_tab.dataset_cleared()
        self.training_tab.clear_dataset()
