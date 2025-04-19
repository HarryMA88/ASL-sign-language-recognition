import os
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

from ml.config import TRAIN_CONFIG
from ml.transforms import get_test_transforms
from backend.webcam_thread import WebcamThread
from frontend.utils.label_map import label_map

class WebcamPopup(QDialog):
    """
    Live webcam prediction dialog.

    Captures frames, detects hand region, applies preprocessing,
    runs inference, and displays predicted probabilities.
    """

    def __init__(self, model, device, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Webcam Prediction")
        self.resize(500, 650)

        self.model = model
        self.model.eval()
        self.device = device
        self.frame = None
        self.crop_coords = None

        # Main UI layout
        self.layout = QVBoxLayout(self)

        # Label to display webcam frames
        self.cam_label = QLabel("Initializing webcam...")
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.cam_label)

        # Button to capture current frame and run prediction
        self.btn_capture = QPushButton("Capture & Predict")
        self.btn_capture.clicked.connect(self.capture_and_predict)
        self.layout.addWidget(self.btn_capture)

        # Label to show predicted class
        self.result_label = QLabel("Prediction: N/A")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.result_label)

        # Set up matplotlib canvas with dark theme
        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.fig.patch.set_facecolor('#2E2E2E')
        self.ax.set_facecolor('#2E2E2E')
        # White text for axes and title
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Start webcam feed in background thread
        self.thread = WebcamThread()
        self.thread.frame_signal.connect(self.update_frame)
        self.thread.start()

    def closeEvent(self, event):
        # Ensure webcam thread stops when dialog closes
        self.thread.stop()
        self.thread.wait()
        event.accept()

    def update_frame(self, frame):
        """
        Receive and display new webcam frame with an overlayed crop box.
        """
        self.frame = frame.copy()
        h, w, _ = frame.shape

        # Define central crop box for hand region
        box_size = 300
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2, y2 = x1 + box_size, y1 + box_size
        self.crop_coords = (x1, y1, x2, y2)

        # Draw red rectangle on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Convert BGR to RGB for Qt
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(400, 400, Qt.KeepAspectRatio)
        self.cam_label.setPixmap(pix)

    def resize_with_padding(self, image, target_size=(28, 28), pad_color=0):
        """
        Resize image to target size while preserving aspect ratio, padding as needed.
        """
        old_h, old_w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / old_w, target_h / old_h)
        new_w, new_h = int(old_w * scale), int(old_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Compute padding on each side
        top = (target_h - new_h) // 2
        bottom = target_h - new_h - top
        left = (target_w - new_w) // 2
        right = target_w - new_w - left

        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=pad_color)

    def capture_and_predict(self):
        """
        Process the current frame: extract hand region, apply preprocessing,
        run inference, and update UI with results.
        """
        if self.frame is None or self.crop_coords is None:
            QMessageBox.warning(self, "Capture Error", "Webcam frame not ready.")
            return

        x1, y1, x2, y2 = self.crop_coords
        roi = self.frame[y1:y2, x1:x2]

        # Apply skin-color mask in HSV space
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Extract largest contour as hand region
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if contours:
            hand_cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(hand_cnt)
            hand = roi[y:y+h, x:x+w]
        else:
            hand = roi

        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Resize and pad using training mean as background
        mean_px = int(TRAIN_CONFIG['normalize_mean'] * 255)
        final = self.resize_with_padding(enhanced, (28,28), pad_color=mean_px)

        # Prepare tensor and run inference
        tensor = get_test_transforms()(final).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy().squeeze()

        # Determine top prediction
        pred_idx = int(output.argmax(1).item())
        pred_label = label_map.get(pred_idx, str(pred_idx))

        # Update probability bar chart
        self.ax.clear()
        bars = self.ax.bar(np.arange(len(probs)), probs, color='orange')
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.title.set_color('white')
        self.ax.set_title('Output Probabilities', color='white')
        self.canvas.draw()

        # Display predicted label
        self.result_label.setText(f"Prediction: {pred_label}")
