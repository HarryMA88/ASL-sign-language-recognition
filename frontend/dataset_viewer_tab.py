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
    Displays dataset thumbnails with label filtering and counts.
    Supports dynamic loading as the user scrolls.
    """
    def __init__(self):
        super().__init__()

        # Apply dark theme and Magistral Light font
        self.setStyleSheet("""
        QWidget { background: #2E2E2E; color: #EEEEEE; }
        QComboBox { background: #444444; color: #EEEEEE; border:none; border-radius:5px; padding:4px; }
        QTableWidget#statsTable {
            background: #2E2E2E; color: #FFFFFF; gridline-color: #444444;
        }
        QHeaderView::section {
            background: #333333; color: #FFFFFF;
            padding: 4px;
            font-family: 'Magistral Light', Arial, sans-serif;
            font-size: 11pt; font-weight: bold; font-style: italic;
        }
        QTableWidget { background: #2E2E2E; gridline-color: #444444; }
        """)

        # Main vertical layout
        main = QVBoxLayout(self)
        main.setContentsMargins(5, 5, 5, 5)
        main.setSpacing(5)

        # Dropdown to filter by label
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        main.addWidget(self.filter_combo)

        # Table showing counts per label
        self.stats_table = QTableWidget()
        self.stats_table.setObjectName("statsTable")
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Label", "Count"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        main.addWidget(self.stats_table)

        # Grid for image thumbnails
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setVisible(False)
        self.table.verticalHeader().setVisible(False)
        main.addWidget(self.table)

        # Internal state
        self.all_data   = []           # list of (image_array, label)
        self.thumb_size = QSize(100, 100)
        self.page_size  = 0            # number of items per batch
        self.max_idx    = 0            # current maximum index to display

        # Load more when scrolling near bottom
        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def dataset_loaded(self, images, labels, shape):
        """Receive new dataset, update filter options, stats, and show initial thumbnails."""
        self.all_data = list(zip(images, labels))
        unique_labels = sorted({lbl for _, lbl in self.all_data})

        # Rebuild filter dropdown
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        for lbl in unique_labels:
            self.filter_combo.addItem(str(lbl))
        self.filter_combo.blockSignals(False)

        # Update stats and paging
        self._populate_stats_table()
        total = len(self.all_data)
        self.page_size = math.ceil(total / 10) if total else 0
        self.max_idx = min(self.page_size, total)

        # Display first batch
        self.refresh_table()

    @pyqtSlot()
    def dataset_cleared(self):
        """Clear all displayed data and reset controls."""
        self.all_data.clear()
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        self.stats_table.clearContents()
        self.stats_table.setRowCount(0)
        self.table.clearContents()
        self.table.setRowCount(0)

    def _populate_stats_table(self):
        """Count labels and populate the stats table."""
        from collections import Counter
        counts = Counter(lbl for _, lbl in self.all_data)
        self.stats_table.setRowCount(len(counts))

        for row, (lbl, cnt) in enumerate(counts.items()):
            item_lbl = QTableWidgetItem(str(lbl))
            item_cnt = QTableWidgetItem(str(cnt))

            # Center text
            for item in (item_lbl, item_cnt):
                item.setTextAlignment(Qt.AlignCenter)
                item.setForeground(QBrush(QColor("#FFFFFF")))
                item.setBackground(QBrush(QColor("#2E2E2E")))
                font = QFont("Magistral Light", 10)
                font.setBold(True)
                item.setFont(font)

            self.stats_table.setItem(row, 0, item_lbl)
            self.stats_table.setItem(row, 1, item_cnt)

    def refresh_table(self):
        """Render thumbnails up to current max index, applying filter."""
        selected = self.filter_combo.currentText()
        if selected == "All":
            data = self.all_data
        else:
            data = [(img, lbl) for img, lbl in self.all_data if str(lbl) == selected]

        disp = data[:self.max_idx]
        if not disp:
            self.table.clearContents()
            self.table.setRowCount(0)
            return

        width = self.table.viewport().width()
        cols = max(1, width // (self.thumb_size.width() + 10))
        rows = math.ceil(len(disp) / cols)
        self.table.setColumnCount(cols)
        self.table.setRowCount(rows)
        self.table.clearContents()

        for idx, (img, lbl) in enumerate(disp):
            r, c = divmod(idx, cols)
            h, w = img.shape
            qimg = QImage((img * 255).astype('uint8').data, w, h, w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(
                self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            label = QLabel()
            label.setPixmap(pix)
            label.setToolTip(str(lbl))
            label.setAlignment(Qt.AlignCenter)

            self.table.setCellWidget(r, c, label)
            self.table.setRowHeight(r, self.thumb_size.height() + 10)

    def _on_scroll(self, value):
        """Load next batch when scrollbar nears the bottom."""
        sb = self.table.verticalScrollBar()
        if value >= sb.maximum() - 10 and self.max_idx < len(self.all_data):
            self.max_idx = min(self.max_idx + self.page_size, len(self.all_data))
            self.refresh_table()

    def resizeEvent(self, event):
        """Re-layout thumbnails when the widget resizes."""
        super().resizeEvent(event)
        self.refresh_table()