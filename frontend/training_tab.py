import time
import torch
import matplotlib.pyplot as plt
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QComboBox, QLineEdit,
    QPushButton, QMessageBox, QProgressBar, QSizePolicy
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from backend.training_thread import TrainingThread
from frontend.timer_widget import TimerWidget

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

        # — Main layout —
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(20)

        # — Top row: parameters and timer/metrics —
        top = QHBoxLayout()
        top.setSpacing(20)
        main.addLayout(top)

        # --- LEFT: hyper-parameters ---
        left = QVBoxLayout()
        top.addLayout(left, stretch=3)

        header = QLabel("Hyper‑parameters")
        header.setObjectName("sectionHeader")
        left.addWidget(header)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 3)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        lbl_split = QLabel("Training/Test Split")
        lbl_split.setObjectName("paramLabel")
        grid.addWidget(lbl_split, 0, 0, 1, 3)

        self.lbl_split_pct = QLabel("80%")
        self.lbl_split_pct.setObjectName("smallLabel")
        grid.addWidget(self.lbl_split_pct, 1, 2, alignment=Qt.AlignRight)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50, 100)
        self.slider.setValue(80)
        grid.addWidget(self.slider, 2, 0, 1, 3)
        self.slider.valueChanged.connect(lambda v: self.lbl_split_pct.setText(f"{v}%"))

        params = ["Model", "Batch Size", "Epochs"]
        for i, txt in enumerate(params):
            lbl = QLabel(txt)
            lbl.setObjectName("paramLabel")
            grid.addWidget(lbl, 3, i)

        self.cb_model = QComboBox()
        self.cb_model.addItems(["Alexnet", "Lebron", "Resnet"])
        grid.addWidget(self.cb_model, 4, 0)

        self.in_batch = QLineEdit()
        self.in_batch.setPlaceholderText("e.g. 32")
        self.in_batch.setValidator(QIntValidator(1, 1024))
        grid.addWidget(self.in_batch, 4, 1)

        self.in_epochs = QLineEdit()
        self.in_epochs.setPlaceholderText("e.g. 30")
        self.in_epochs.setValidator(QIntValidator(1, 999))
        grid.addWidget(self.in_epochs, 4, 2)

        left.addLayout(grid)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop  = QPushButton("Stop Training")
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        left.addLayout(btn_layout)

        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop.clicked.connect(self.stop_training)

        # --- RIGHT: timer and metrics —
        right = QVBoxLayout()
        right.setSpacing(4)
        top.addLayout(right, stretch=2)

        lbl_time = QLabel("Time Elapsed")
        lbl_time.setObjectName("smallLabel")
        lbl_time.setContentsMargins(0, 10, 0, 0)
        right.addWidget(lbl_time, alignment=Qt.AlignCenter)

        self.timer_widget = TimerWidget()
        right.addWidget(self.timer_widget, alignment=Qt.AlignCenter)

        metric_row = QHBoxLayout()
        metric_row.setSpacing(10)
        right.addLayout(metric_row)

        for title, attr in [("Train Loss", "l_val"), ("Val Accuracy", "a_val")]:
            col = QVBoxLayout(); col.setSpacing(4)
            lbl = QLabel(title); lbl.setObjectName("smallLabel")
            val = QLabel("0%");    val.setObjectName("valueLabel")
            setattr(self, attr, val)
            col.addWidget(lbl, alignment=Qt.AlignCenter)
            col.addWidget(val, alignment=Qt.AlignCenter)
            metric_row.addLayout(col)

        # — Matplotlib canvas —
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.patch.set_facecolor('#2E2E2E')
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background: #2E2E2E;")
        self.ax_loss, self.ax_acc = axes
        for ax in (self.ax_loss, self.ax_acc):
            ax.set_facecolor('#2E2E2E')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
        # initial titles
        self.ax_loss.set_title("Training Loss", color='white')
        self.ax_acc.set_title("Validation Accuracy", color='white')
        main.addWidget(self.canvas)

        # — Epoch label —
        self.lbl_epoch = QLabel("Epoch 0/0")
        self.lbl_epoch.setObjectName("smallLabel")
        main.addWidget(self.lbl_epoch, alignment=Qt.AlignCenter)

        # — Progress bar container & bar (initially hidden) —
        progress_container = QHBoxLayout()
        progress_container.addStretch()
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.hide()
        progress_container.addWidget(self.progress_bar)
        progress_container.addStretch()
        main.addLayout(progress_container)

        # — State initialization —
        self.losses  = []
        self.accs    = []
        self.thread  = None
        self.dataset = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Make progress bar half the widget's width
        self.progress_bar.setFixedWidth(self.width() // 2)

    def set_dataset(self, d):
        self.dataset = d

    def clear_dataset(self):
        self.dataset = None
        self.in_batch.clear()
        self.in_epochs.clear()
        self.lbl_epoch.setText("Epoch 0/0")
        self.progress_bar.hide()
        self.timer_widget.setText("00:00")
        self.l_val.setText("0%")
        self.a_val.setText("0%")
        self.losses.clear()
        self.accs.clear()
        self.canvas.draw()

    def start_training(self):
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "No dataset.")
            return
        try:
            bs = int(self.in_batch.text())
            ne = int(self.in_epochs.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Enter valid batch & epochs.")
            return

        # Reset UI
        self.losses.clear()
        self.accs.clear()
        self.lbl_epoch.setText(f"Epoch 0/{ne}")
        self.progress_bar.setMaximum(ne)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        tp = self.slider.value() / 100
        mc = self.cb_model.currentText()

        self.thread = TrainingThread(mc, self.dataset, tp, bs, ne)
        self.thread.started.connect(self.timer_widget.start)
        self.thread.finished_signal.connect(self.timer_widget.stop)
        self.btn_stop.clicked.connect(self.timer_widget.stop)

        self.thread.epoch_signal.connect(self._on_epoch)
        self.thread.finished_signal.connect(self._on_done)
        self.thread.error_signal.connect(self._on_error)
        self.thread.start()

    @pyqtSlot(int, float, float, float)
    def _on_epoch(self, e, loss, acc, elapsed):
        # Update labels
        self.l_val.setText(f"{loss:.1f}%")
        self.a_val.setText(f"{acc*100:.1f}%")
        self.lbl_epoch.setText(f"Epoch {e}/{self.thread.num_epochs}")
        # Update progress bar
        self.progress_bar.setValue(e)
        # Update plots
        self.losses.append(loss)
        self.accs.append(acc)

        self.ax_loss.clear()
        self.ax_loss.set_title("Training Loss", color='white')
        self.ax_loss.plot(range(1, e+1), self.losses, color='orange')

        self.ax_acc.clear()
        self.ax_acc.set_title("Validation Accuracy", color='white')
        self.ax_acc.plot(range(1, e+1), self.accs, color='green')

        self.canvas.draw()

    @pyqtSlot(dict)
    def _on_done(self, meta):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.timer_widget.stop()
        self.progress_bar.hide()

        if len(meta.get("train_loss", [])) < self.thread.num_epochs:
            QMessageBox.information(self, "Stopped", "Training was halted before completion.")
        else:
            QMessageBox.information(self, "Done", "Training complete.")
            self.training_finished.emit(self.thread.model, self.thread.metadata)


    @pyqtSlot(str)
    def _on_error(self, msg):
        self.timer_widget.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Error", msg)

    def stop_training(self):
        if self.thread:
            self.thread.stop()
            self.timer_widget.stop()
            self.btn_stop.setEnabled(False)
