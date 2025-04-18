# frontend/training_tab.py

import time
import torch
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIntValidator, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QComboBox, QLineEdit,
    QPushButton, QProgressBar, QMessageBox, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from backend.training_thread import TrainingThread

class TrainingTab(QWidget):
    training_finished = pyqtSignal(object, dict)

    def __init__(self):
        super().__init__()

        # —— Dark theme & Magistral Light font —— 
        self.setStyleSheet("""
        QWidget { background-color: #2E2E2E; color: #EEEEEE;
                   font-family: 'Magistral Light', Arial, sans-serif; }
        QLabel#sectionHeader { font-size: 18pt; font-weight: bold; }
        QLabel#smallLabel   { font-size: 10pt; color: #CCCCCC; }
        QLabel#valueLabel   { font-size: 24pt; font-weight: bold; color: #FFFFFF; }
        QSlider::groove:horizontal { background: #444; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: orange; width: 14px; margin: -4px 0; border-radius: 7px; }
        QLabel#paramLabel   { font-size: 12pt; font-weight: bold; color: #FFFFFF; }
        QComboBox, QLineEdit {
            background: #444444; color: #FFFFFF;
            border: none; border-radius: 4px;
            padding: 4px; font-size: 11pt;
        }
        QPushButton {
            background-color: #444444; color: #EEEEEE; border:none;
            padding: 6px 12px; border-radius: 5px; font-weight: bold;
        }
        QPushButton:disabled { background-color: #333333; color: #777777; }
        QPushButton:hover:!disabled { background-color: #555555; }
        QProgressBar {
            background: #444444; border: none; border-radius: 5px;
            height: 12px; color: #EEEEEE; text-align: center;
        }
        QProgressBar::chunk { background: orange; border-radius:5px; }
        """)

        # — Main vertical layout —
        main = QVBoxLayout(self)
        main.setContentsMargins(12,12,12,12)
        main.setSpacing(20)

        # — Top row: two columns —
        top = QHBoxLayout()
        top.setSpacing(40)
        main.addLayout(top)

        # --- LEFT: hyper-parameters ---
        left = QVBoxLayout()
        top.addLayout(left, stretch=3)

        # Section header
        hdr = QLabel("Hyper‑parameters")
        hdr.setObjectName("sectionHeader")
        left.addWidget(hdr)

        # Grid for controls
        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)

        # Training/Test Split
        lbl_split = QLabel("Training/Test Split")
        lbl_split.setObjectName("paramLabel")
        grid.addWidget(lbl_split, 0, 0, 1, 2)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50,100)
        self.slider.setValue(80)
        grid.addWidget(self.slider, 1, 0)
        self.lbl_split_pct = QLabel(f"{self.slider.value()}%")
        self.lbl_split_pct.setObjectName("smallLabel")
        grid.addWidget(self.lbl_split_pct, 1, 1, alignment=Qt.AlignLeft)
        self.slider.valueChanged.connect(lambda v: self.lbl_split_pct.setText(f"{v}%"))

        # Model / Batch / Epochs labels
        for i, txt in enumerate(["Model", "Batch Size", "Epochs"]):
            lbl = QLabel(txt); lbl.setObjectName("paramLabel")
            grid.addWidget(lbl, 2, i)

        # Inputs
        self.cb_model = QComboBox()
        self.cb_model.addItems(["Alexnet","Lebron","Resnet"])
        grid.addWidget(self.cb_model, 3, 0)
        self.in_batch = QLineEdit(); self.in_batch.setPlaceholderText("e.g. 32")
        self.in_batch.setValidator(QIntValidator(1,1024))
        grid.addWidget(self.in_batch, 3, 1)
        self.in_epochs= QLineEdit(); self.in_epochs.setPlaceholderText("e.g. 30")
        self.in_epochs.setValidator(QIntValidator(1,999))
        grid.addWidget(self.in_epochs, 3, 2)

        left.addLayout(grid)

        # Start/Stop buttons
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop  = QPushButton("Stop Training")
        self.btn_stop.setEnabled(False)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        left.addLayout(btns)

        # Connect
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop.clicked.connect(self.stop_training)

        # — RIGHT: time & metrics —
        right = QVBoxLayout()
        top.addLayout(right, stretch=2)

        # Time Elapsed label + value
        lbl_time = QLabel("Time Elapsed")
        lbl_time.setObjectName("smallLabel")
        right.addWidget(lbl_time, alignment=Qt.AlignCenter)
        self.lbl_timer = QLabel("00 : 00")
        self.lbl_timer.setObjectName("valueLabel")
        right.addWidget(self.lbl_timer, alignment=Qt.AlignCenter)

        # Spacer
        right.addSpacing(10)

        # Train Loss / Val Acc
        metric_row = QHBoxLayout()
        right.addLayout(metric_row)
        # Loss
        col1 = QVBoxLayout()
        l_lbl = QLabel("Train Loss"); l_lbl.setObjectName("smallLabel")
        self.l_val = QLabel("0%");    self.l_val.setObjectName("valueLabel")
        col1.addWidget(l_lbl, alignment=Qt.AlignCenter)
        col1.addWidget(self.l_val, alignment=Qt.AlignCenter)
        metric_row.addLayout(col1)
        # Accuracy
        col2 = QVBoxLayout()
        a_lbl = QLabel("Val. Accuracy"); a_lbl.setObjectName("smallLabel")
        self.a_val = QLabel("0%");    self.a_val.setObjectName("valueLabel")
        col2.addWidget(a_lbl, alignment=Qt.AlignCenter)
        col2.addWidget(self.a_val, alignment=Qt.AlignCenter)
        metric_row.addLayout(col2)

        # — Middle: graph area —
        fig, axes = plt.subplots(1, 2, figsize=(8,4))
        fig.patch.set_facecolor('#2E2E2E')
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background: #2E2E2E;")
        self.ax_loss, self.ax_acc = axes
        for ax in axes:
            ax.set_facecolor('#2E2E2E')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        main.addWidget(self.canvas)

        # Epoch label under graphs
        self.lbl_epoch = QLabel("Epoch 0/0")
        self.lbl_epoch.setObjectName("smallLabel")
        main.addWidget(self.lbl_epoch, alignment=Qt.AlignLeft)

        # — Internal state —
        self.losses = []; self.accs = []
        self.thread = None; self.dataset = None

    def set_dataset(self, d):
        self.dataset = d

    def clear_dataset(self):
        self.dataset = None
        self.in_batch.clear(); self.in_epochs.clear()
        self.lbl_epoch.setText("Epoch 0/0")
        self.lbl_timer.setText("00 : 00")
        self.l_val.setText("0%"); self.a_val.setText("0%")
        self.losses.clear(); self.accs.clear()
        self.canvas.draw()

    def start_training(self):
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "No dataset.")
            return
        # Validate inputs
        try:
            bs = int(self.in_batch.text()); ne = int(self.in_epochs.text())
        except:
            QMessageBox.warning(self, "Warning", "Enter valid batch & epochs.")
            return

        # Setup
        self.losses.clear(); self.accs.clear()
        self.lbl_epoch.setText(f"Epoch 0/{ne}")
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

        tp = self.slider.value()/100
        mc = self.cb_model.currentText()

        self.thread = TrainingThread(mc, self.dataset, tp, bs, ne)
        self.thread.epoch_signal.connect(self._on_epoch)
        self.thread.finished_signal.connect(self._on_done)
        self.thread.error_signal.connect(self._on_error)
        self.thread.start_time = time.time()
        self.thread.start()

    @pyqtSlot(int, float, float, float)
    def _on_epoch(self, e, loss, acc, elapsed):
        # update timer
        mins = int(elapsed//60); secs=int(elapsed%60)
        self.lbl_timer.setText(f"{mins:02d} : {secs:02d}")
        # update loss/acc
        self.l_val.setText(f"{loss:.1f}%")
        self.a_val.setText(f"{acc*100:.1f}%")
        # update epoch label
        total = self.thread.num_epochs
        self.lbl_epoch.setText(f"Epoch {e}/{total}")
        # redraw plots
        self.losses.append(loss); self.accs.append(acc)
        self.ax_loss.clear(); self.ax_loss.set_facecolor('#2E2E2E')
        self.ax_loss.plot(range(1,e+1), self.losses, color='orange')
        self.ax_acc.clear(); self.ax_acc.set_facecolor('#2E2E2E')
        self.ax_acc.plot(range(1,e+1), self.accs, color='orange')
        self.canvas.draw()

    @pyqtSlot(dict)
    def _on_done(self, meta):
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "Done", "Training complete.")
        self.training_finished.emit(self.thread.model, self.thread.metadata)

    @pyqtSlot(str)
    def _on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def stop_training(self):
        if self.thread:
            self.thread.stop()
            self.btn_stop.setEnabled(False)
            QMessageBox.information(self, "Stopped", "Training halted.")
