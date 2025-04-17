import time
import torch
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox, QSpinBox, QPushButton, QProgressBar, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from backend.training_thread import TrainingThread

class TrainingTab(QWidget):
    training_finished = pyqtSignal(object, dict)

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        # slider
        h1=QHBoxLayout(); h1.addWidget(QLabel("Training %:"))
        self.slider_train=QSlider(Qt.Horizontal); self.slider_train.setRange(50,100); self.slider_train.setValue(80)
        h1.addWidget(self.slider_train)
        self.lbl_pct=QLabel("80%"); h1.addWidget(self.lbl_pct)
        self.slider_train.valueChanged.connect(lambda v:self.lbl_pct.setText(f"{v}%"))
        self.layout.addLayout(h1)
        # model
        h2=QHBoxLayout(); h2.addWidget(QLabel("Model:"))
        self.model_combo=QComboBox(); self.model_combo.addItems(["Alexnet","Lebron","Resnet"]);
        h2.addWidget(self.model_combo); self.layout.addLayout(h2)
        # batch+epochs
        h3=QHBoxLayout(); h3.addWidget(QLabel("Batch:"))
        self.spin_batch=QSpinBox(); self.spin_batch.setRange(1,999); self.spin_batch.setValue(32)
        h3.addWidget(self.spin_batch); h3.addWidget(QLabel("Epochs:"))
        self.spin_epochs=QSpinBox(); self.spin_epochs.setRange(1,999); self.spin_epochs.setValue(30)
        h3.addWidget(self.spin_epochs); self.layout.addLayout(h3)
        # start/stop
        h4=QHBoxLayout();
        self.btn_start=QPushButton("Start"); self.btn_start.clicked.connect(self.start_training)
        h4.addWidget(self.btn_start)
        self.btn_stop=QPushButton("Stop"); self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self.stop_training)
        h4.addWidget(self.btn_stop)
        self.layout.addLayout(h4)
        # progress
        self.lbl_progress=QLabel("Progress: N/A"); self.layout.addWidget(self.lbl_progress)
        self.progress_bar=QProgressBar(); self.progress_bar.setRange(0,self.spin_epochs.value());
        self.layout.addWidget(self.progress_bar)
        self.spin_epochs.valueChanged.connect(lambda v:self.progress_bar.setMaximum(v))
        # plot
        self.fig,self.ax=plt.subplots(1,2,figsize=(8,4))
        self.canvas=FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)
        self.losses=[]; self.accs=[]
        self.thread=None; self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_dataset(self,d): self.dataset=d
    def clear_dataset(self): self.dataset=None; QMessageBox.information(self,"Cleared","Dataset removed.")

    def start_training(self):
        if not getattr(self,'dataset',None):
            QMessageBox.warning(self,"Warn","No dataset.");return
        self.losses.clear(); self.accs.clear(); self.progress_bar.setValue(0)
        self.btn_start.setEnabled(False);self.btn_stop.setEnabled(True)
        tp=self.slider_train.value()/100;mc=self.model_combo.currentText();bs=self.spin_batch.value();ne=self.spin_epochs.value()
        self.thread=TrainingThread(mc,self.dataset,tp,bs,ne,self.device)
        self.thread.epoch_signal.connect(self.update_training_progress)
        self.thread.finished_signal.connect(self.finished)
        self.thread.error_signal.connect(self.error)
        self.thread.start_time=time.time();self.thread.start()

    @pyqtSlot(int,float,float,float)
    def update_training_progress(self,e,l,a,el):
        self.losses.append(l); self.accs.append(a)
        self.lbl_progress.setText(f"Epoch {e}/{self.spin_epochs.value()}: Loss={l:.4f} Acc={a*100:.2f}% Elapsed={int(el)}s")
        self.progress_bar.setValue(e)
        self.ax[0].clear(); self.ax[0].plot(range(1,e+1),self.losses,label="Loss"); self.ax[0].legend()
        self.ax[1].clear(); self.ax[1].plot(range(1,e+1),self.accs,label="Acc"); self.ax[1].legend()
        self.canvas.draw()

    @pyqtSlot(dict)
    def finished(self,metrics):
        self.btn_start.setEnabled(True);self.btn_stop.setEnabled(False)
        QMessageBox.information(self,"Done","Training complete.")
        self.training_finished.emit(self.thread.model,self.thread.metadata)

    @pyqtSlot(str)
    def error(self,m):
        QMessageBox.critical(self,"Err",m);self.btn_start.setEnabled(True);self.btn_stop.setEnabled(False)

    def stop_training(self):
        if self.thread: self.thread.stop();self.btn_stop.setEnabled(False);self.lbl_progress.setText("Stopped.")