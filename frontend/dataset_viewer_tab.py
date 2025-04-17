import math
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel, QTableWidget, QAbstractItemView
from PyQt5.QtGui import QImage, QPixmap

class DatasetViewerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        self.layout.addWidget(self.filter_combo)
        self.stats_label = QLabel()
        self.layout.addWidget(self.stats_label)
        self.table = QTableWidget()
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layout.addWidget(self.table)
        self.all_data = []
        self.thumb = QSize(100,100)
        self.page_size = 0
        self.max_idx = 0
        self.table.verticalScrollBar().valueChanged.connect(self.check_scroll)

    def load_dataset(self, images, labels):
        self.all_data = list(zip(images, labels))
        uniq = sorted({l for _,l in self.all_data})
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear(); self.filter_combo.addItem("All")
        for u in uniq: self.filter_combo.addItem(str(u))
        self.filter_combo.blockSignals(False)
        self.update_stats()
        tot = len(self.all_data)
        self.page_size = math.ceil(tot/10) if tot else 0
        self.max_idx = min(self.page_size, tot)
        self.refresh_table()

    def update_stats(self):
        from collections import Counter
        cnt = Counter(l for _,l in self.all_data)
        txt = "Dataset Statistics:\n" + "\n".join(f"{k}: {v}" for k,v in cnt.items())
        self.stats_label.setText(txt if cnt else "No data")

    def refresh_table(self):
        filt = self.filter_combo.currentText()
        data = (self.all_data if filt=="All" else [(i,l) for i,l in self.all_data if str(l)==filt])
        disp = data[:self.max_idx]
        if not disp:
            self.table.clearContents(); self.table.setRowCount(0); return
        w = self.table.viewport().width()
        cols = max(1, w//(self.thumb.width()+10))
        rows = (len(disp)+cols-1)//cols
        self.table.setColumnCount(cols); self.table.setRowCount(rows)
        self.table.horizontalHeader().setVisible(False); self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False); self.table.clearContents()
        for idx, (img,lbl) in enumerate(disp):
            r, c = divmod(idx, cols)
            h,w = img.shape
            qimg = QImage((img*255).astype('uint8').data, w,h,w, QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg).scaled(self.thumb, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lblw = QLabel(); lblw.setPixmap(pix); lblw.setToolTip(str(lbl)); lblw.setAlignment(Qt.AlignCenter)
            self.table.setCellWidget(r,c,lblw)
        for r in range(rows): self.table.setRowHeight(r, self.thumb.height()+10)

    def check_scroll(self, val):
        sb = self.table.verticalScrollBar()
        if val >= sb.maximum()-10 and self.max_idx < len(self.all_data):
            self.max_idx = min(self.max_idx + self.page_size, len(self.all_data))
            self.refresh_table()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.refresh_table()