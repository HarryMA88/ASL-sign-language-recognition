import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5.QtCore import pyqtSlot
from .dataset_import_tab import DatasetImportTab
from .dataset_viewer_tab import DatasetViewerTab
from .training_tab import TrainingTab
from .prediction_tab import PredictionTab
from backend.dataset import SignLanguageDataset

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition Tool")
        self.resize(1000,800)
        self.tabs=QTabWidget(); self.setCentralWidget(self.tabs)
        self.import_tab=DatasetImportTab()
        self.viewer_tab=DatasetViewerTab()
        self.training_tab=TrainingTab()
        self.prediction_tab=PredictionTab()
        self.tabs.addTab(self.import_tab,"Import")
        self.tabs.addTab(self.viewer_tab,"Viewer")
        self.tabs.addTab(self.training_tab,"Training")
        self.tabs.addTab(self.prediction_tab,"Prediction")
        self.tabs.setTabEnabled(self.tabs.indexOf(self.prediction_tab),False)
        self.import_tab.dataset_loaded.connect(self.on_dataset_loaded)
        self.import_tab.dataset_cleared.connect(self.on_dataset_cleared)
        self.training_tab.training_finished.connect(self.on_training_finished)

    @pyqtSlot(np.ndarray,np.ndarray,tuple)
    def on_dataset_loaded(self, imgs, lbls, shape):
        self.dataset=SignLanguageDataset(imgs,lbls)
        self.viewer_tab.load_dataset(imgs,lbls)
        self.training_tab.set_dataset(self.dataset)
    @pyqtSlot()
    def on_dataset_cleared(self):
        self.dataset=None
        self.viewer_tab.load_dataset(np.array([]),np.array([]))
        self.training_tab.clear_dataset()
    @pyqtSlot(object,dict)
    def on_training_finished(self,m,md):
        self.prediction_tab.set_model(m,md)
        self.tabs.setTabEnabled(self.tabs.indexOf(self.prediction_tab),True)

def main():
    app=QApplication(sys.argv)
    w=MainWindow();w.show();sys.exit(app.exec_())

if __name__=="__main__": main()