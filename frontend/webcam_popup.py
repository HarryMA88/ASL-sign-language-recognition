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
        self.crop_coords = None

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

        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

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
        box_size = 300  # increased red-box to 300×300

        x_center = int(w * 0.65)
        y_center = h // 2
        x1 = x_center - box_size // 2
        y1 = y_center - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        self.crop_coords = (x1, y1, x2, y2)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)
        )

    @staticmethod
    def extract_hand_region(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 30, 60], dtype=np.uint8)
        upper = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        hand = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(hand)
        return img[y:y+h, x:x+w]

    def capture_and_predict(self):
        if self.frame is None or self.crop_coords is None:
            QMessageBox.warning(self, "Missing Data", "Webcam frame or crop box not ready.")
            return

        x1, y1, x2, y2 = self.crop_coords
        roi = self.frame[y1:y2, x1:x2]

        # optional: refine with skin segmentation
        # roi = self.extract_hand_region(roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28))
        img_np = resized.astype(np.uint8)

        # apply transforms and debug
        tensor = get_test_transforms()(img_np)
        print(f"[DEBUG] tensor.shape={tensor.shape}, "
              f"min={tensor.min():.3f}, max={tensor.max():.3f}, "
              f"mean={tensor.mean():.3f}, std={tensor.std():.3f}")

        batch = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(batch)
            probs = torch.nn.functional.softmax(out, dim=1).squeeze().cpu().numpy()
            pred_idx = int(out.argmax(1).item())

        label = label_map.get(pred_idx, str(pred_idx))
        self.result_label.setText(f"Prediction: {label}")

        self.ax.clear()
        self.ax.bar(np.arange(len(probs)), probs)
        self.ax.set_title("Output Probabilities")
        self.ax.set_xlabel("Class")
        self.ax.set_ylabel("Probability")
        self.canvas.draw()
