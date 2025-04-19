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
    A tab that shows your dataset: lets you filter by label, peek at label counts,
    and endlessly scroll through thumbnails.

    Hooks:
      - dataset_loaded(images, labels, shape): load fresh data
      - dataset_cleared(): clear everything out
    """
    def __init__(self):
        super().__init__()

        # Slap on a dark theme & Magistral Light font
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

        # Main layout box with padding
        main = QVBoxLayout(self)
        main.setContentsMargins(5, 5, 5, 5)
        main.setSpacing(5)

        # Add a filter dropdown up top
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        main.addWidget(self.filter_combo)

        # Show label counts in a little stats table
        self.stats_table = QTableWidget()
        self.stats_table.setObjectName("statsTable")
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Label", "Count"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        main.addWidget(self.stats_table)

        # Grid for thumbnail previews
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        main.addWidget(self.table)

        # Tidy up internal vars
        self.all_data = []           # holds (image, label) pairs
        self.thumb_size = QSize(100, 100)
        self.page_size = 0           # how many to load per batch
        self.max_idx = 0             # current end index

        # Endless scrolling setup
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def dataset_loaded(self, images, labels, shape):
        """Load new data, rebuild filter options and stats, then show first batch."""
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
        self.page_size = math.ceil(total / 10) if total else 0
        self.max_idx = min(self.page_size, total)
        self.refresh_table()

    @pyqtSlot()
    def dataset_cleared(self):
        """Wipe everything clean back to the start state."""
        self.all_data.clear()
        self.filter_combo.clear(); self.filter_combo.addItem("All")
        self.stats_table.clearContents(); self.stats_table.setRowCount(0)
        self.table.clearContents(); self.table.setRowCount(0)

    def _populate_stats_table(self):
        """Tally up labels and show counts."""
        from collections import Counter
        cnt = Counter(lbl for _, lbl in self.all_data)
        self.stats_table.setRowCount(len(cnt))

        for row, (label, count) in enumerate(cnt.items()):
            item_label = QTableWidgetItem(str(label))
            item_count = QTableWidgetItem(str(count))

            # Centre text nicely
            item_label.setTextAlignment(Qt.AlignCenter)
            item_count.setTextAlignment(Qt.AlignCenter)

            # Bright-white text
            brush = QBrush(QColor("#FFFFFF"))
            item_label.setForeground(brush)
            item_count.setForeground(brush)

            # Dark grey lil’ background
            bg = QBrush(QColor("#2E2E2E"))
            item_label.setBackground(bg)
            item_count.setBackground(bg)

            # Give it some bold flair
            cell_font = QFont("Magistral Light", 10)
            cell_font.setBold(True)
            item_label.setFont(cell_font)
            item_count.setFont(cell_font)

            self.stats_table.setItem(row, 0, item_label)
            self.stats_table.setItem(row, 1, item_count)

    def refresh_table(self):
        """Redraw the thumbnail grid based on the current filter and scroll position."""
        sel = self.filter_combo.currentText()
        data = self.all_data if sel == "All" else [
            (img, lbl) for img, lbl in self.all_data if str(lbl) == sel
        ]

        disp_max = min(self.max_idx, len(data))
        disp = data[:disp_max]

        if not disp:
            self.table.clearContents()
            self.table.setRowCount(0)
            return

        w = self.table.viewport().width()
        cols = max(1, w // (self.thumb_size.width() + 10))
        rows = math.ceil(len(disp) / cols)
        self.table.setColumnCount(cols)
        self.table.setRowCount(rows)

        self.table.clearContents()
        
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
        """When you scroll near the bottom, load the next batch of thumbnails."""
        sb = self.table.verticalScrollBar()
        if val >= sb.maximum() - 10 and self.max_idx < len(self.all_data):
            self.max_idx = min(self.max_idx + self.page_size, len(self.all_data))
            self.refresh_table()

    def resizeEvent(self, event):
        """Tidy up the grid when the window resizes."""
        super().resizeEvent(event)
        self.refresh_table()
