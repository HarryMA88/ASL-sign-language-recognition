import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import os
import time

from ml.config import TRAIN_CONFIG
from ml.transforms import get_test_transforms
from backend.webcam_thread import WebcamThread
from frontend.utils.label_map import label_map

class WebcamPopup(QDialog):
    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Webcam Prediction")
        self.resize(400, 550)

        self.model = model
        self.model.eval()
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
        box_size = 300

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

    def resize_with_padding(self, image, target_size=(28, 28), pad_color=0):
        old_h, old_w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / old_w, target_h / old_h)
        new_w, new_h = int(old_w * scale), int(old_h * scale)
        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        return cv2.copyMakeBorder(
            resized_image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_color
        )

    def capture_and_predict(self):
        if self.frame is None or self.crop_coords is None:
            QMessageBox.warning(self, "Missing Data", "Webcam frame or crop box not ready.")
            return

        # 1️⃣ Raw crop of the red box
        x1, y1, x2, y2 = self.crop_coords
        roi = self.frame[y1:y2, x1:x2]

        # 2️⃣ Skin‑color mask (in HSV)
        hsv   = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        mask  = cv2.inRange(hsv, lower, upper)
        # clean it up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # 3️⃣ Find largest contour & crop
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            hx, hy, hw, hh = cv2.boundingRect(c)
            hand = roi[hy:hy+hh, hx:hx+hw]
        else:
            hand = roi

        # 4️⃣ (Debug) save mask & hand crop so you can inspect them
        cv2.imwrite("DEBUG_mask.png", mask)
        cv2.imwrite("DEBUG_hand.png", hand)

        # 5️⃣ Grayscale + CLAHE on the hand crop
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)

        # 6️⃣ Resize + pad with train‑mean background
        mean_px = int(TRAIN_CONFIG["normalize_mean"] * 255)
        resized = self.resize_with_padding(gray_eq, (28,28), pad_color=mean_px)

        # 7️⃣ (Optional) inspect the final input
        cv2.imwrite("DEBUG_resized.png", resized)

        # 8️⃣ Transform & batch exactly as before
        tensor = get_test_transforms()(resized)
        batch  = tensor.unsqueeze(0).to(self.device)

        # 9️⃣ Inference
        with torch.no_grad():
            out   = self.model(batch)
            probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
        top5 = np.argsort(probs)[::-1][:5]
        print("DEBUG top‑5 (idx,prob):", [(int(i), float(probs[i])) for i in top5])

        idx    = int(out.argmax(1).item())
        label  = label_map.get(idx, str(idx))
        print(f"DEBUG argmax idx={idx} → {label}")

        # 🔟 Update UI
        self.result_label.setText(f"Prediction: {label}")
        self.ax.clear()
        self.ax.bar(np.arange(len(probs)), probs)
        self.ax.set_title("Output Probabilities")
        self.ax.set_xlabel("Class")
        self.ax.set_ylabel("Probability")
        self.canvas.draw()




