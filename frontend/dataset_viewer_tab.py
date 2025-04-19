import math
import numpy as np
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout,
    QComboBox, QLabel, QTableWidget, QTableWidgetItem, QAbstractItemView
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QBrush, QColor

class DatasetViewerTab(QWidget):
    """
    Tab for displaying datasets: supports filtering, stats in a table, and infinite scroll.
    Listens to signals:
      - dataset_loaded(images, labels, shape): loads new data
      - dataset_cleared(): clears view
    """
    def __init__(self):
        super().__init__()

        # Dark theme & Magistral Light styling
        self.setStyleSheet("""
        QWidget { background: #2E2E2E; color: #EEEEEE; }
        QComboBox { background: #444444; color: #EEEEEE;
                    border: none; border-radius:5px; padding:4px; }
        /* Stats table styling */
        QTableWidget#statsTable {
            background-color: #2E2E2E;
            color: #FFFFFF;
            gridline-color: #444444;
        }
        QHeaderView::section {
            background-color: #333333;
            color: #FFFFFF;
            padding: 4px;
            font-family: 'Magistral Light', Arial, sans-serif;
            font-size: 11pt;
            font-weight: bold;
            font-style: italic;
        }
        QTableWidget { background: #2E2E2E; gridline-color: #444444; }
        """)

        # Main layout
        main = QVBoxLayout(self)
        main.setContentsMargins(5,5,5,5)
        main.setSpacing(5)

        # Filter dropdown
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        main.addWidget(self.filter_combo)

        # Statistics table
        self.stats_table = QTableWidget()
        self.stats_table.setObjectName("statsTable")
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Label", "Count"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        main.addWidget(self.stats_table)

        # Table for thumbnails
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        main.addWidget(self.table)

        # Internal state
        self.all_data = []           # list of (image, label)
        self.thumb_size = QSize(100,100)
        self.page_size = 0
        self.max_idx = 0

        # Infinite scroll
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def dataset_loaded(self, images, labels, shape):
        """Slot to receive loaded data."""
        self.all_data = list(zip(images, labels))
        uniq = sorted({lbl for _, lbl in self.all_data})
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        for u in uniq:
            self.filter_combo.addItem(str(u))
        self.filter_combo.blockSignals(False)

        self._populate_stats_table()
        total = len(self.all_data)
        self.page_size = math.ceil(total/10) if total else 0
        self.max_idx = min(self.page_size, total)
        self.refresh_table()

    @pyqtSlot()
    def dataset_cleared(self):
        self.all_data.clear()
        self.filter_combo.clear(); self.filter_combo.addItem("All")
        self.stats_table.clearContents(); self.stats_table.setRowCount(0)
        self.table.clearContents(); self.table.setRowCount(0)

    def _populate_stats_table(self):
        from collections import Counter
        cnt = Counter(lbl for _, lbl in self.all_data)
        self.stats_table.setRowCount(len(cnt))

        for row, (label, count) in enumerate(cnt.items()):
            item_label = QTableWidgetItem(str(label))
            item_count = QTableWidgetItem(str(count))

            # Center text
            item_label.setTextAlignment(Qt.AlignCenter)
            item_count.setTextAlignment(Qt.AlignCenter)

            # White text
            brush = QBrush(QColor("#FFFFFF"))
            item_label.setForeground(brush)
            item_count.setForeground(brush)

            # Dark-grey cell background
            bg = QBrush(QColor("#2E2E2E"))
            item_label.setBackground(bg)
            item_count.setBackground(bg)

            # Bold font
            cell_font = QFont("Magistral Light", 10)
            cell_font.setBold(True)
            item_label.setFont(cell_font)
            item_count.setFont(cell_font)

            self.stats_table.setItem(row, 0, item_label)
            self.stats_table.setItem(row, 1, item_count)

    def refresh_table(self):
        sel = self.filter_combo.currentText()
        data = self.all_data if sel == "All" else [
            (img, lbl) for img, lbl in self.all_data if str(lbl) == sel
        ]
        disp = data[:self.max_idx]
        if not disp:
            self.table.clearContents(); self.table.setRowCount(0); return

        w = self.table.viewport().width()
        cols = max(1, w // (self.thumb_size.width() + 10))
        rows = math.ceil(len(disp) / cols)
        self.table.setColumnCount(cols); self.table.setRowCount(rows)

        for idx, (img, lbl) in enumerate(disp):
            r, c = divmod(idx, cols)
            h, w = img.shape
            qimg = QImage((img * 255).astype('uint8').data, w, h, w,
                          QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(
                self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            cell = QLabel()
            cell.setPixmap(pix)
            cell.setToolTip(str(lbl))
            cell.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(r, c, cell)
            self.table.setRowHeight(r, self.thumb_size.height() + 10)

    def _on_scroll(self, val):
        sb = self.table.verticalScrollBar()
        if val >= sb.maximum() - 10 and self.max_idx < len(self.all_data):
            self.max_idx = min(self.max_idx + self.page_size, len(self.all_data))
            self.refresh_table()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_table()
