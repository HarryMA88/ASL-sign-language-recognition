import sys
import os
import time
import json
import math
import shutil
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from backend.webcam_thread import WebcamThread

# PyQt5 imports
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QComboBox, QScrollArea,
    QSlider, QSpinBox, QMessageBox, QListWidget, QListWidgetItem, QTableWidget,
    QTableWidgetItem, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QSize
from PyQt5.QtGui import QImage, QPixmap

# Matplotlib for live plotting embedded in PyQt
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class PredictionTab(QWidget):
    def __init__(self):
        super(PredictionTab, self).__init__()
        self.layout = QVBoxLayout(self)
        
        self.btn_load_model = QPushButton("Load Saved Model")
        self.btn_load_model.clicked.connect(self.load_model)
        self.layout.addWidget(self.btn_load_model)
        
        self.list_images = QListWidget()
        self.layout.addWidget(self.list_images)
        
        self.btn_load_images = QPushButton("Load Test Images")
        self.btn_load_images.clicked.connect(self.load_test_images)
        self.layout.addWidget(self.btn_load_images)
        
        self.btn_predict = QPushButton("Run Prediction on Selected Images")
        self.btn_predict.clicked.connect(self.run_prediction)
        self.layout.addWidget(self.btn_predict)
        
        self.lbl_prediction = QLabel("Prediction: N/A")
        self.layout.addWidget(self.lbl_prediction)
        
        self.btn_start_webcam = QPushButton("Start Webcam")
        self.btn_start_webcam.clicked.connect(self.start_webcam)
        self.layout.addWidget(self.btn_start_webcam)
        
        self.btn_stop_webcam = QPushButton("Stop Webcam")
        self.btn_stop_webcam.clicked.connect(self.stop_webcam)
        self.btn_stop_webcam.setEnabled(False)
        self.layout.addWidget(self.btn_stop_webcam)
        
        self.btn_screenshot = QPushButton("Take Screenshot")
        self.btn_screenshot.clicked.connect(self.take_screenshot)
        self.layout.addWidget(self.btn_screenshot)
        
        self.webcam_label = QLabel()
        self.layout.addWidget(self.webcam_label)
        
        self.webcam_thread = None
        self.current_frame = None
        self.loaded_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Saved Model", "", "Model Files (*.pt)")
        if not file_path:
            return
        try:
            checkpoint = torch.load(file_path, map_location=self.device)
            self.loaded_model = checkpoint["model_state"]
            self.model_metadata = checkpoint["metadata"]
            model_choice = self.model_metadata["model_choice"]
            input_shape = self.model_metadata.get("input_shape", (28, 28))
            num_classes = 36
            if model_choice == "Alexnet":
                model = CustomCNN(input_shape, num_classes)
            elif model_choice == "Lebron":
                model = StandardCNN1(input_shape, num_classes)
            elif model_choice == "Resnet":
                model = StandardCNN2(input_shape, num_classes)
            else:
                QMessageBox.warning(self, "Warning", "Unknown model architecture in metadata.")
                return
            model.load_state_dict(self.loaded_model)
            model.to(self.device)
            model.eval()
            self.loaded_model = model
            QMessageBox.information(self, "Model Loaded", "Model loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
    
    def load_test_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if not files:
            return
        self.list_images.clear()
        for file in files:
            item = QListWidgetItem(file)
            self.list_images.addItem(item)
    
    def run_prediction(self):
        if self.loaded_model is None:
            QMessageBox.warning(self, "Warning", "No model loaded.")
            return
        selected_items = self.list_images.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No images selected.")
            return
        predictions = []
        for item in selected_items:
            file_path = item.text()
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            input_shape = self.model_metadata.get("input_shape", (28, 28))
            img = cv2.resize(img, input_shape)
            img = img.astype(np.float32) / 255.0
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            output = self.loaded_model(img_tensor)
            _, pred = torch.max(output, 1)
            pred_val = pred.item()
            if pred_val < 26:
                label = chr(pred_val + ord('A'))
            else:
                label = str(pred_val - 26)
            predictions.append(label)
        self.lbl_prediction.setText("Predictions: " + ", ".join(predictions))
    
    def start_webcam(self):
        self.webcam_thread = WebcamThread()
        self.webcam_thread.frame_signal.connect(self.update_webcam_frame)
        self.webcam_thread.start()
        self.btn_start_webcam.setEnabled(False)
        self.btn_stop_webcam.setEnabled(True)
    
    def stop_webcam(self):
        if self.webcam_thread:
            self.webcam_thread.stop()
            self.webcam_thread = None
        self.btn_start_webcam.setEnabled(True)
        self.btn_stop_webcam.setEnabled(False)
        self.webcam_label.clear()
    
    @pyqtSlot(np.ndarray)
    def update_webcam_frame(self, frame):
        self.current_frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(320, 240, Qt.KeepAspectRatio)
        self.webcam_label.setPixmap(pixmap)
    
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

        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                          cv2.BORDER_CONSTANT, value=pad_color)
        return padded_image
    
    def take_screenshot(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "No webcam frame to capture.")
            return
        
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        resized_gray = self.resize_with_padding(gray, target_size=(28, 28), pad_color=0)

        screenshot_dir = "screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(screenshot_dir, f"screenshot_{timestamp}.png")
        cv2.imwrite(filename, resized_gray)
        QMessageBox.information(self, "Screenshot", f"Screenshot saved as {filename}")
        item = QListWidgetItem(filename)
        self.list_images.addItem(item)