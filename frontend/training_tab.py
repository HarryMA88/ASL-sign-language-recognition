# frontend/dataset_viewer_tab.py

import math
import numpy as np
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QComboBox, QLabel,
    QTableWidget, QTableWidgetItem, QAbstractItemView
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QBrush, QColor

class DatasetViewerTab(QWidget):
    """
    Displays dataset thumbnails with label filtering and count stats.
    Supports incremental loading on scroll.
    """
    def __init__(self):
        super().__init__()

        # Dark theme + Magistral Light font
        self.setStyleSheet("""
            QWidget { background: #2E2E2E; color: #EEEEEE; }
            QComboBox { background: #444; color: #EEE; border:none; padding:4px; }
            QTableWidget#statsTable { background: #2E2E2E; color: #FFF; gridline-color:#444; }
            QHeaderView::section {
                background: #333; color: #FFF; padding:4px;
                font-family:'Magistral Light'; font-size:11pt;
                font-weight:bold; font-style:italic;
            }
            QTableWidget { background: #2E2E2E; gridline-color:#444; }
        """)

        # Layout setup
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5,5,5,5)
        layout.setSpacing(5)

        # Label filter
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        layout.addWidget(self.filter_combo)

        # Stats table
        self.stats_table = QTableWidget(objectName="statsTable",
                                        columnCount=2,
                                        horizontalHeaderLabels=["Label","Count"])
        self.stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.stats_table)

        # Thumbnail grid
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # State
        self.all_data = []         # (image_array, label)
        self.thumb_size = QSize(100,100)
        self.page_size = 0         # items per batch
        self.max_idx = 0           # current max index

        # Scroll-trigger
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def dataset_loaded(self, images, labels, shape):
        """Load data, update filters & stats, and show first batch."""
        self.all_data = list(zip(images, labels))
        labels_set = sorted({lbl for _, lbl in self.all_data})

        # Rebuild filter options
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        for lbl in labels_set:
            self.filter_combo.addItem(str(lbl))
        self.filter_combo.blockSignals(False)

        # Update stats
        self._populate_stats()
        total = len(self.all_data)
        self.page_size = math.ceil(total / 10) if total else 0
        self.max_idx = min(self.page_size, total)

        self.refresh_table()

    @pyqtSlot()
    def dataset_cleared(self):
        """Reset all views to empty."""
        self.all_data.clear()
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        self.stats_table.clearContents()
        self.stats_table.setRowCount(0)
        self.table.clearContents()
        self.table.setRowCount(0)

    def _populate_stats(self):
        """Count labels and fill the stats table."""
        from collections import Counter
        counts = Counter(lbl for _, lbl in self.all_data)
        self.stats_table.setRowCount(len(counts))
        font = QFont("Magistral Light", 10, QFont.Bold)

        for row, (lbl, cnt) in enumerate(counts.items()):
            item_lbl = QTableWidgetItem(str(lbl))
            item_cnt = QTableWidgetItem(str(cnt))
            for item in (item_lbl, item_cnt):
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QBrush(QColor("#FFF")))
                item.setBackground(QBrush(QColor("#2E2E2E")))
                item.setFont(font)
            self.stats_table.setItem(row, 0, item_lbl)
            self.stats_table.setItem(row, 1, item_cnt)

    def refresh_table(self):
        """Render thumbnails up to current max index with active filter."""
        sel = self.filter_combo.currentText()
        data = self.all_data if sel=="All" else [(img,lbl) for img,lbl in self.all_data if str(lbl)==sel]
        disp = data[:self.max_idx]

        if not disp:
            self.table.clearContents()
            self.table.setRowCount(0)
            return

        width = self.table.viewport().width()
        cols = max(1, width // (self.thumb_size.width()+10))
        rows = math.ceil(len(disp)/cols)
        self.table.setColumnCount(cols)
        self.table.setRowCount(rows)
        self.table.clearContents()

        for idx, (img, lbl) in enumerate(disp):
            r, c = divmod(idx, cols)
            h, w = img.shape
            qimg = QImage((img*255).astype('uint8').data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lblw = QLabel()
            lblw.setPixmap(pix)
            lblw.setToolTip(str(lbl))
            lblw.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(r, c, lblw)
            self.table.setRowHeight(r, self.thumb_size.height()+10)

    def _on_scroll(self, val):
        """Load next batch when scrollbar nears bottom."""
        sb = self.table.verticalScrollBar()
        if val >= sb.maximum()-10 and self.max_idx < len(self.all_data):
            self.max_idx = min(self.max_idx + self.page_size, len(self.all_data))
            self.refresh_table()

    def resizeEvent(self, event):
        """Reflow thumbnails on resize."""
        super().resizeEvent(event)
        self.refresh_table()

# frontend/training_tab.py

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
    Configures and monitors model training:
    - Set split ratio, model, batch size, epochs
    - Displays elapsed time, loss, accuracy, and live plots
    """
    training_finished = pyqtSignal(object, dict)

    def __init__(self):
        super().__init__()
        # Apply dark theme
        self.setStyleSheet("""
            QWidget { background:#2E2E2E; color:#EEE; font-family:'Magistral Light'; }
            QLabel#sectionHeader { font-size:18pt; font-weight:bold; }
            QLabel#smallLabel   { font-size:10pt; color:#CCC; }
            QLabel#valueLabel   { font-size:24pt; font-weight:bold; color:#FFF; }
            QSlider::groove:horizontal { background:#444; height:6px; border-radius:3px; }
            QSlider::handle:horizontal { background:orange; width:14px; margin:-4px; border-radius:7px; }
            QLabel#paramLabel { font-size:12pt; font-weight:bold; color:#FFF; }
            QComboBox,QLineEdit {
                background:#444; color:#FFF; border:none; border-radius:4px; padding:4px; font-size:11pt;
            }
            QPushButton { background:#444; color:#EEE; border:none; padding:6px 12px; border-radius:5px; font-weight:bold; }
            QPushButton:disabled { background:#333; color:#777; }
            QPushButton:hover:!disabled { background:#555; }
            QProgressBar { background:#444; border:none; border-radius:5px; height:12px; }
            QProgressBar::chunk { background:orange; border-radius:5px; }
        """)

        main = QVBoxLayout(self)
        main.setContentsMargins(12,12,12,12)
        main.setSpacing(20)

        # Top row: params on left, timer & metrics on right
        top = QHBoxLayout(); top.setSpacing(20)
        main.addLayout(top)

        # LEFT: hyper-parameters
        left = QVBoxLayout()
        top.addLayout(left, stretch=3)
        header = QLabel("Hyper‑parameters", objectName="sectionHeader")
        left.addWidget(header)

        grid = QGridLayout(); grid.setSpacing(10)
        grid.setColumnStretch(0,3); grid.setColumnStretch(1,1); grid.setColumnStretch(2,1)
        # Split ratio
        lbl_split = QLabel("Training/Test Split", objectName="paramLabel")
        grid.addWidget(lbl_split, 0,0,1,3)
        self.lbl_split_pct = QLabel("80%", objectName="smallLabel")
        grid.addWidget(self.lbl_split_pct, 1,2, alignment=Qt.AlignRight)
        self.slider = QSlider(Qt.Horizontal, value=80, minimum=50, maximum=100)
        grid.addWidget(self.slider, 2,0,1,3)
        self.slider.valueChanged.connect(lambda v: self.lbl_split_pct.setText(f"{v}%"))
        # Model, batch, epochs
        labels = ["Model","Batch Size","Epochs"]
        for i, txt in enumerate(labels):
            grid.addWidget(QLabel(txt, objectName="paramLabel"), 3,i)
        self.cb_model = QComboBox(); self.cb_model.addItems(["Alexnet","Lebron","Resnet"])
        grid.addWidget(self.cb_model, 4,0)
        self.in_batch = QLineEdit(validator=QIntValidator(1,1024), placeholderText="e.g. 32")
        grid.addWidget(self.in_batch, 4,1)
        self.in_epochs= QLineEdit(validator=QIntValidator(1,999),  placeholderText="e.g. 30")
        grid.addWidget(self.in_epochs, 4,2)
        left.addLayout(grid)

        # Start/Stop
        btns = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop  = QPushButton("Stop Training", enabled=False)
        btns.addWidget(self.btn_start); btns.addWidget(self.btn_stop)
        left.addLayout(btns)
        self.btn_start.clicked.connect(self.start_training)
        self.btn_stop.clicked.connect(self.stop_training)

        # RIGHT: timer & metrics
        right = QVBoxLayout(); right.setSpacing(4)
        top.addLayout(right, stretch=2)
        right.addWidget(QLabel("Time Elapsed", objectName="smallLabel"), alignment=Qt.AlignCenter)
        self.timer_widget = TimerWidget()
        right.addWidget(self.timer_widget, alignment=Qt.AlignCenter)
        metrics = QHBoxLayout(); metrics.setSpacing(10)
        for title, attr in [("Train Loss","l_val"),("Val Accuracy","a_val")]:
            col = QVBoxLayout(); col.setSpacing(4)
            col.addWidget(QLabel(title, objectName="smallLabel"), alignment=Qt.AlignCenter)
            lbl = QLabel("0%", objectName="valueLabel")
            setattr(self, attr, lbl)
            col.addWidget(lbl, alignment=Qt.AlignCenter)
            metrics.addLayout(col)
        right.addLayout(metrics)

        # Plot canvas
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(8,4))
        fig.patch.set_facecolor('#2E2E2E')
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background:#2E2E2E;")
        for ax in (ax1,ax2):
            ax.set_facecolor('#2E2E2E')
            ax.tick_params(colors='white')
            ax.title.set_color('white')
        self.ax_loss, self.ax_acc = ax1,ax2
        self.ax_loss.set_title("Training Loss", color='white')
        self.ax_acc.set_title("Validation Accuracy", color='white')
        main.addWidget(self.canvas)

        # Epoch label
        self.lbl_epoch = QLabel("Epoch 0/0", objectName="smallLabel")
        main.addWidget(self.lbl_epoch, alignment=Qt.AlignCenter)

        # Progress bar
        prog_layout = QHBoxLayout()
        prog_layout.addStretch()
        self.progress_bar = QProgressBar(textVisible=False, minimum=0, maximum=0,
                                         sizePolicy=QSizePolicy.Expanding,QSizePolicy.Fixed)
        self.progress_bar.hide()
        prog_layout.addWidget(self.progress_bar)
        prog_layout.addStretch()
        main.addLayout(prog_layout)

        # State
        self.losses = []; self.accs = []
        self.thread = None; self.dataset = None

    def resizeEvent(self, event):
        """Adjust progress bar width on resize."""
        super().resizeEvent(event)
        self.progress_bar.setFixedWidth(self.width() // 2)

    def set_dataset(self, d):
        """Assign new dataset for training."""
        self.dataset = d

    def clear_dataset(self):
        """Reset UI after dataset removal."""
        self.dataset = None
        for fld in (self.in_batch, self.in_epochs):
            fld.clear()
        self.lbl_epoch.setText("Epoch 0/0")
        self.progress_bar.hide()
        self.timer_widget.reset()
        for lbl in (self.l_val, self.a_val):
            lbl.setText("0%")
        self.losses.clear(); self.accs.clear()
        self.canvas.draw()

    def start_training(self):
        """Validate inputs, start training thread, and initialize UI."""
        if not self.dataset:
            QMessageBox.warning(self, "Warning", "No dataset.")
            return
        try:
            bs = int(self.in_batch.text())
            ne = int(self.in_epochs.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Enter valid batch & epochs.")
            return

        # Prepare UI
        self.losses.clear(); self.accs.clear()
        self.lbl_epoch.setText(f"Epoch 0/{ne}")
        self.progress_bar.setMaximum(ne); self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

        # Start thread
        tp = self.slider.value() / 100
        mc = self.cb_model.currentText()
        self.thread = TrainingThread(mc, self.dataset, tp, bs, ne)
        self.thread.started.connect(self.timer_widget.start)
        self.thread.finished_signal.connect(self.timer_widget.stop)
        self.thread.epoch_signal.connect(self._on_epoch)
        self.thread.finished_signal.connect(self._on_done)
        self.thread.error_signal.connect(self._on_error)
        self.thread.start()

    @pyqtSlot(int, float, float, float)
    def _on_epoch(self, e, loss, acc, elapsed):
        """Update progress, metrics, and plots each epoch."""
        self.l_val.setText(f"{loss:.1f}%")
        self.a_val.setText(f"{acc*100:.1f}%")
        self.lbl_epoch.setText(f"Epoch {e}/{self.thread.num_epochs}")
        self.progress_bar.setValue(e)

        self.losses.append(loss); self.accs.append(acc)
        self.ax_loss.clear();    self.ax_loss.plot(self.losses, color='orange')
        self.ax_acc.clear();     self.ax_acc.plot(self.accs,  color='green')
        self.ax_loss.set_title("Training Loss", color='white')
        self.ax_acc.set_title("Validation Accuracy", color='white')
        self.canvas.draw()

    @pyqtSlot(dict)
    def _on_done(self, meta):
        """Cleanup after training finishes."""
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.timer_widget.stop(); self.progress_bar.hide()
        QMessageBox.information(self, "Done", "Training complete.")
        self.training_finished.emit(self.thread.model, self.thread.metadata)

    @pyqtSlot(str)
    def _on_error(self, msg):
        """Handle training errors."""
        self.timer_widget.stop()
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)
        self.progress_bar.hide()
        QMessageBox.critical(self, "Error", msg)

    def stop_training(self):
        """Stop the training thread prematurely."""
        if self.thread:
            self.thread.stop()
            self.timer_widget.stop()
            self.btn_stop.setEnabled(False)
            QMessageBox.information(self, "Stopped", "Training halted.")