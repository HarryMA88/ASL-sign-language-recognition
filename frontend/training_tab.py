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
    """
    Tab for configuring and running model training.

    Features:
      - Adjust split ratio, model choice, batch size, and epochs
      - Live timer, loss, and accuracy metrics
      - Real-time plotting of training curves
    """

    training_finished = pyqtSignal(object, dict)

    def __init__(self):
        super().__init__()

        # Dark theme and font styling
        self.setStyleSheet(
            """
            QWidget { background-color: #2E2E2E; color: #EEEEEE; font-family: 'Magistral Light', Arial, sans-serif; }
            QLabel#sectionHeader { font-size: 18pt; font-weight: bold; }
            QLabel#smallLabel   { font-size: 10pt; color: #CCCCCC; }
            QLabel#valueLabel   { font-size: 24pt; font-weight: bold; }
            QSlider::groove      { background: #444; height: 6px; border-radius: 3px; }
            QSlider::handle      { background: orange; width: 14px; border-radius: 7px; }
            QLabel#paramLabel    { font-size: 12pt; font-weight: bold; }
            QComboBox, QLineEdit { background: #444; color: #FFF; border: none; border-radius: 4px; padding: 4px; }
            QPushButton          { background: #444; color: #EEE; padding: 6px 12px; border-radius: 5px; }
            QPushButton:hover:!disabled { background: #555; }
            QProgressBar         { background: #444; border-radius: 5px; color: #EEE; text-align: center; }
            QProgressBar::chunk  { background: orange; }
            """
        )

        # Main vertical layout
        main = QVBoxLayout(self)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(20)

        # Top row: controls (left) and metrics (right)
        top = QHBoxLayout()
        main.addLayout(top)

        # --- Left pane: hyper-parameters ---
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

        # Split slider
        split_lbl = QLabel("Training/Test Split")
        split_lbl.setObjectName("paramLabel")
        grid.addWidget(split_lbl, 0, 0, 1, 3)

        self.split_val = QLabel("80%")
        self.split_val.setObjectName("smallLabel")
        grid.addWidget(self.split_val, 1, 2, alignment=Qt.AlignRight)

        self.split_slider = QSlider(Qt.Horizontal)
        self.split_slider.setRange(50, 100)
        self.split_slider.setValue(80)
        self.split_slider.valueChanged.connect(
            lambda v: self.split_val.setText(f"{v}%")
        )
        grid.addWidget(self.split_slider, 2, 0, 1, 3)

        # Model, batch size, epochs
        labels = ["Model", "Batch Size", "Epochs"]
        for idx, text in enumerate(labels):
            lbl = QLabel(text)
            lbl.setObjectName("paramLabel")
            grid.addWidget(lbl, 3, idx)

        self.cb_model = QComboBox()
        self.cb_model.addItems(["Alexnet", "Lebron", "Resnet"])
        grid.addWidget(self.cb_model, 4, 0)

        self.input_batch = QLineEdit(placeholderText="16–256")
        self.input_batch.setValidator(QIntValidator(16, 256))
        grid.addWidget(self.input_batch, 4, 1)

        self.input_epochs = QLineEdit(placeholderText="10–100")
        self.input_epochs.setValidator(QIntValidator(10, 100))
        grid.addWidget(self.input_epochs, 4, 2)

        left.addLayout(grid)

        # Start/Stop buttons
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setEnabled(False)
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)
        left.addLayout(btns)

        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop.clicked.connect(self.stop_training)

        # --- Right pane: timer and metrics ---
        right = QVBoxLayout()
        right.setSpacing(4)
        top.addLayout(right, stretch=2)

        time_lbl = QLabel("Time Elapsed")
        time_lbl.setObjectName("smallLabel")
        right.addWidget(time_lbl, alignment=Qt.AlignCenter)

        self.timer = TimerWidget()
        right.addWidget(self.timer, alignment=Qt.AlignCenter)

        metrics = QHBoxLayout(); right.addLayout(metrics)
        for title, attr in [("Train Loss", "l_val"), ("Val Accuracy", "a_val")]:
            col = QVBoxLayout(); col.setSpacing(4)
            m_lbl = QLabel(title); m_lbl.setObjectName("smallLabel")
            m_val = QLabel("0%"); m_val.setObjectName("valueLabel")
            setattr(self, attr, m_val)
            col.addWidget(m_lbl, alignment=Qt.AlignCenter)
            col.addWidget(m_val, alignment=Qt.AlignCenter)
            metrics.addLayout(col)

        # --- Plot canvas ---
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        fig.patch.set_facecolor('#2E2E2E')
        self.canvas = FigureCanvas(fig)
        main.addWidget(self.canvas)
        self.ax_loss, self.ax_acc = axs
        for ax in axs:
            ax.set_facecolor('#2E2E2E')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')
        self.ax_loss.set_title("Training Loss", color='white')
        self.ax_acc.set_title("Validation Accuracy", color='white')

        # Epoch label
        self.lbl_epoch = QLabel("Epoch 0/0")
        self.lbl_epoch.setObjectName("smallLabel")
        main.addWidget(self.lbl_epoch, alignment=Qt.AlignCenter)

        # --- Progress bar ---
        pb_container = QHBoxLayout()
        pb_container.addStretch()
        self.progress_bar = QProgressBar(textVisible=False)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.progress_bar.hide()
        pb_container.addWidget(self.progress_bar)
        pb_container.addStretch()
        main.addLayout(pb_container)

        # State initialization
        self.losses = []
        self.accs = []
        self.thread = None
        self.dataset = None

    def resizeEvent(self, event):
        """Keep progress bar proportional on resize."""
        super().resizeEvent(event)
        self.progress_bar.setFixedWidth(self.width() // 2)

    @pyqtSlot(object)
    def set_dataset(self, data):
        """Assign the dataset object for training."""
        self.dataset = data

    @pyqtSlot()
    def clear_dataset(self):
        """Clear training inputs and reset metrics."""
        self.dataset = None
        self.input_batch.clear()
        self.input_epochs.clear()
        self.lbl_epoch.setText("Epoch 0/0")
        self.progress_bar.hide()
        self.timer.setText("00:00")
        self.l_val.setText("0%")
        self.a_val.setText("0%")
        self.losses.clear()
        self.accs.clear()
        self.canvas.draw()

    def start_training(self):
        """Validate inputs, reset UI, and start training thread."""
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return
        try:
            bs = int(self.input_batch.text())
            ne = int(self.input_epochs.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Enter valid numbers.")
            return
        if not (16 <= bs <= 256):
            QMessageBox.warning(self, "Warning", "Batch size must be 16–256.")
            return
        if not (10 <= ne <= 100):
            QMessageBox.warning(self, "Warning", "Epochs must be 10–100.")
            return

        # UI reset for new training run
        self.losses.clear()
        self.accs.clear()
        self.lbl_epoch.setText(f"Epoch 0/{ne}")
        self.progress_bar.setMaximum(ne)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

        # Launch training thread
        split = self.split_slider.value() / 100
        model_choice = self.cb_model.currentText()
        self.thread = TrainingThread(model_choice, self.dataset, split, bs, ne)
        self.thread.started.connect(self.timer.start)
        self.thread.epoch_signal.connect(self._on_epoch)
        self.thread.finished_signal.connect(self._on_done)
        self.thread.error_signal.connect(self._on_error)
        self.thread.finished_signal.connect(self.timer.stop)
        self.btn_stop.clicked.connect(self.timer.stop)
        self.thread.start()

    @pyqtSlot(int, float, float, float)
    def _on_epoch(self, epoch, loss, acc, elapsed):
        """Update UI each epoch with new metrics and plots."""
        self.l_val.setText(f"{loss:.1f}%")
        self.a_val.setText(f"{acc*100:.1f}%")
        self.lbl_epoch.setText(f"Epoch {epoch}/{self.thread.num_epochs}")
        self.progress_bar.setValue(epoch)

        self.losses.append(loss)
        self.accs.append(acc)

        self.ax_loss.clear()
        self.ax_loss.set_title("Training Loss", color='white')
        self.ax_loss.plot(range(1, epoch+1), self.losses, color='white')

        self.ax_acc.clear()
        self.ax_acc.set_title("Validation Accuracy", color='white')
        self.ax_acc.plot(range(1, epoch+1), self.accs, color='white')

        self.canvas.draw()

    @pyqtSlot(dict)
    def _on_done(self, meta):
        """Restore UI when training completes."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.hide()
        self.timer.stop()
        QMessageBox.information(self, "Done", "Training complete.")
        self.training_finished.emit(self.thread.model, self.thread.metadata)

    @pyqtSlot(str)
    def _on_error(self, msg):
        """Handle errors from the training thread."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.hide()
        self.timer.stop()
        QMessageBox.critical(self, "Error", msg)

    def stop_training(self):
        """Stop the training process prematurely."""
        if self.thread:
            self.thread.stop()
            self.timer.stop()
            self.btn_stop.setEnabled(False)
            QMessageBox.information(self, "Stopped", "Training halted.")
