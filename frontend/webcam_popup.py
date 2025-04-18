# frontend/webcam_popup.py

import time
import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from ml.transforms import get_test_transforms
from backend.webcam_thread import WebcamThread
from frontend.utils.label_map import label_map

class WebcamPopup(QDialog):
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Webcam Prediction")
        self.resize(400, 550)

        self.model = model
        self.device = device
        self.frame = None

        self.layout = QVBoxLayout(self)
        self.cam_label = QLabel("Starting webcam...")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.cam_label)

        self.btn_capture = QPushButton("Capture and Predict")
        self.btn_capture.clicked.connect(self.capture_and_predict)
        self.layout.addWidget(self.btn_capture)

        self.result_label = QLabel("Prediction: N/A")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        # ——— Matplotlib canvas with dark-grey bg ———
        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        # Figure background
        self.fig.patch.set_facecolor('#2E2E2E')
        # Axes background & text styling
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background-color: #2E2E2E;")
        self.layout.addWidget(self.canvas)

        # Start the webcam thread
        self.thread = WebcamThread()
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    def closeEvent(self, event):
        self.thread.stop()
        self.thread.wait()
        event.accept()

    def update_frame(self, frame):
        self.frame = frame.copy()

        h, w, _ = frame.shape
        box_size = 200
        x_center = int(w * 0.65)
        y_center = h // 2
        x1 = x_center - box_size // 2
        y1 = y_center - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)
        )

        self.crop_coords = (x1, y1, x2, y2)

    def capture_and_predict(self):
        if self.frame is None:
            return

        x1, y1, x2, y2 = getattr(self, "crop_coords", (None,)*4)
        if None in (x1, y1, x2, y2):
            QMessageBox.warning(self, "Missing Crop Box", "Crop box not initialized.")
            return

        roi = self.frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        img_np = resized.astype(np.uint8)

        tensor = get_test_transforms()(img_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1).squeeze().cpu().numpy()
            pred_idx = int(out.argmax(1).item())

        label = label_map.get(pred_idx, f"{pred_idx}")
        self.result_label.setText(f"Prediction: {label}")

        # ——— redraw bar chart with orange bars and white text ———
        self.ax.clear()
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')

        # Plot orange bars
        x = np.arange(len(probs))
        self.ax.bar(x, probs, color='orange')
        self.ax.set_title("Output Probabilities")
        self.ax.set_xlabel("Class")
        self.ax.set_ylabel("Probability")

        self.canvas.draw()
