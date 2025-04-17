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

#####################################
# PyTorch Dataset and Model Classes #
#####################################

class SignLanguageDataset(Dataset):
    """
    A custom dataset class that holds image data and labels.
    Images are expected to be NumPy arrays (grayscale) and labels as characters.
    """
    def __init__(self, images, labels):
        self.images = images  # shape: (N, H, W)
        self.labels = labels  # shape: (N,)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        # Convert the image (H,W) to a tensor with shape (1, H, W)
        image = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)
        label = int(self.labels[idx])
        return image, label

#####################################
# ImportThread for CSV Import
#####################################

class ImportThread(QThread):
    """
    A thread to import the dataset from a CSV file.
    Emits progress updates (with a rough ETA) and returns images, labels, and image shape.
    Assumes that the CSV file has the label in the first column and flattened pixel values in the remaining columns.
    Rows with labels "J" and "Z" are skipped.
    """
    progress_signal = pyqtSignal(int, str)  # progress value (0-100) and ETA string
    finished_signal = pyqtSignal(np.ndarray, np.ndarray, tuple)  # images, labels, image shape

    def __init__(self, file_path):
        super(ImportThread, self).__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):
        try:
            start_time = time.time()
            df_iter = pd.read_csv(self.file_path, chunksize=1000)
            images_list = []
            labels_list = []
            total_rows = 0
            for chunk in df_iter:
                if not self._is_running:
                    break
                for _, row in chunk.iterrows():
                    if not self._is_running:
                        break
                    label = row.iloc[0]
                    if label in ['J', 'Z']:
                        continue
                    pixels = row.iloc[1:].values.astype(np.float32) / 255.0
                    num_pixels = len(pixels)
                    img_dim = int(np.sqrt(num_pixels))
                    if img_dim * img_dim != num_pixels:
                        continue  # skip non-square rows
                    image = pixels.reshape(img_dim, img_dim)
                    images_list.append(image)
                    labels_list.append(label)
                    total_rows += 1
                    elapsed = time.time() - start_time
                    progress = int((total_rows / 10000) * 100)
                    remaining = int((elapsed / total_rows) * (10000 - total_rows)) if total_rows > 0 else 0
                    eta = f"{remaining // 60} Min, {remaining % 60} Sec left"
                    self.progress_signal.emit(min(progress, 100), eta)
            if self._is_running:
                images = np.array(images_list)
                labels = np.array(labels_list)
                img_shape = images[0].shape if len(images) > 0 else (0, 0)
                self.finished_signal.emit(images, labels, img_shape)
        except Exception as e:
            print("Error during import:", e)

    def stop(self):
        self._is_running = False

#####################################
# TrainingThread for Model Training
#####################################

class TrainingThread(QThread):
    """
    A thread to train the selected CNN model.
    It splits the provided dataset into training and validation sets, trains the model for a number of epochs,
    and emits epoch-by-epoch updates with training loss, validation accuracy, and elapsed time.
    """
    epoch_signal = pyqtSignal(int, float, float, float)  # epoch, train_loss, val_acc, elapsed_time
    finished_signal = pyqtSignal(dict)  # final metrics (as a dictionary)
    error_signal = pyqtSignal(str)

    def __init__(self, model_choice, dataset, train_percentage, batch_size, num_epochs, device):
        super(TrainingThread, self).__init__()
        self.model_choice = model_choice
        self.dataset = dataset
        self.train_percentage = train_percentage
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device
        self._is_running = True
        self.metadata = {}

    def run(self):
        try:
            total = len(self.dataset)
            train_size = int(self.train_percentage * total)
            val_size = total - train_size
            train_set, val_set = random_split(self.dataset, [train_size, val_size])
            train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
            sample_img, _ = self.dataset[0]
            input_shape = sample_img.shape[1:]
            num_classes = 36  # 26 letters + 10 digits

            if self.model_choice == "Lebron":
                model = CustomCNN(input_shape, num_classes)
            elif self.model_choice == "Alexnet":
                model = StandardCNN1(input_shape, num_classes)
            elif self.model_choice == "Resnet":
                model = StandardCNN2(input_shape, num_classes)
            else:
                self.error_signal.emit("Unknown model selected.")
                return

            model.to(self.device)
            optimizer = optim.Adam(model.parameters())
            criterion = nn.CrossEntropyLoss()
            start_time = time.time()
            metrics = {"train_loss": [], "val_acc": []}

            for epoch in range(1, self.num_epochs + 1):
                if not self._is_running:
                    break
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    if not self._is_running:
                        break
                    images = images.to(self.device)
                    targets = []
                    for label in labels:
                        if label.isdigit():
                            targets.append(int(label) + 26)
                        else:
                            targets.append(ord(label.upper()) - ord('A'))
                    targets = torch.tensor(targets, dtype=torch.long).to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)
                epoch_loss = running_loss / train_size
                metrics["train_loss"].append(epoch_loss)

                model.eval()
                correct = 0
                total_val = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(self.device)
                        targets = []
                        for label in labels:
                            if label.isdigit():
                                targets.append(int(label) + 26)
                            else:
                                targets.append(ord(label.upper()) - ord('A'))
                        targets = torch.tensor(targets, dtype=torch.long).to(self.device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total_val += targets.size(0)
                        correct += (predicted == targets).sum().item()
                val_acc = correct / total_val if total_val > 0 else 0.0
                metrics["val_acc"].append(val_acc)
                elapsed_time = time.time() - start_time
                self.epoch_signal.emit(epoch, epoch_loss, val_acc, elapsed_time)
            self.finished_signal.emit(metrics)
            self.metadata = {
                "model_choice": self.model_choice,
                "train_percentage": self.train_percentage,
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "train_loss": metrics["train_loss"],
                "val_acc": metrics["val_acc"],
                "elapsed_time": elapsed_time,
                "input_shape": input_shape
            }
            self.model = model
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

#####################################
# WebcamThread for Live Webcam Feed
#####################################

class WebcamThread(QThread):
    """
    A thread to capture frames from the webcam using OpenCV.
    Emits each frame as a NumPy array.
    """
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super(WebcamThread, self).__init__()
        self._is_running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        while self._is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_signal.emit(frame)
            time.sleep(0.03)
        self.cap.release()

    def stop(self):
        self._is_running = False

#####################################
# PyQt GUI: Tab Widgets and Main Window
#####################################

# -------------------- Dataset Import Tab -------------------- #
class DatasetImportTab(QWidget):
    """
    The Dataset Import tab allows the user to select a CSV file.
    The chosen CSV file is copied to the 'data' folder.
    Only one CSV is allowed at a time.
    Once finished, the tab emits the loaded images, labels, and image shape.
    """
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()

    def __init__(self):
        super(DatasetImportTab, self).__init__()
        self.layout = QVBoxLayout(self)
        self.btn_import = QPushButton("Import Dataset CSV")
        self.btn_import.clicked.connect(self.import_dataset)
        self.layout.addWidget(self.btn_import)
        self.btn_remove = QPushButton("Remove Imported Dataset")
        self.btn_remove.clicked.connect(self.remove_dataset)
        self.layout.addWidget(self.btn_remove)
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.lbl_eta = QLabel("ETA: N/A")
        self.layout.addWidget(self.lbl_eta)
        self.btn_stop = QPushButton("Stop Import")
        self.btn_stop.clicked.connect(self.stop_import)
        self.btn_stop.setEnabled(False)
        self.layout.addWidget(self.btn_stop)
        self.import_thread = None

        self.data_folder = "data"
        if not os.path.isdir(self.data_folder):
            os.makedirs(self.data_folder)
        self.update_buttons_state()

    def update_buttons_state(self):
        csv_files = [f for f in os.listdir(self.data_folder) if f.lower().endswith(".csv")]
        if len(csv_files) == 0:
            self.btn_import.setEnabled(True)
            self.btn_remove.setEnabled(False)
        elif len(csv_files) == 1:
            self.btn_import.setEnabled(False)
            self.btn_remove.setEnabled(True)
        else:
            self.btn_import.setEnabled(False)
            self.btn_remove.setEnabled(True)

    def import_dataset(self):
        csv_files = [f for f in os.listdir(self.data_folder) if f.lower().endswith(".csv")]
        if len(csv_files) > 0:
            QMessageBox.warning(self, "Warning", "A CSV file already exists in the 'data' folder. Please remove it before importing a new one.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        file_name = os.path.basename(file_path)
        new_path = os.path.join(self.data_folder, file_name)
        try:
            shutil.copy(file_path, new_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy file to data folder: {e}")
            return
        self.btn_import.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.import_thread = ImportThread(new_path)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.import_finished)
        self.import_thread.start()
        self.update_buttons_state()

    @pyqtSlot(int, str)
    def update_progress(self, value, eta):
        self.progress_bar.setValue(value)
        self.lbl_eta.setText(f"ETA: {eta}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def import_finished(self, images, labels, img_shape):
        self.btn_import.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_eta.setText("Import completed")
        self.dataset_loaded.emit(images, labels, img_shape)

    def stop_import(self):
        if self.import_thread:
            self.import_thread.stop()
            self.btn_stop.setEnabled(False)
            self.lbl_eta.setText("Import stopped")

    def remove_dataset(self):
        csv_files = [f for f in os.listdir(self.data_folder) if f.lower().endswith(".csv")]
        for csv_file in csv_files:
            file_path = os.path.join(self.data_folder, csv_file)
            try:
                os.remove(file_path)
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to remove {csv_file}: {e}")
        self.progress_bar.setValue(0)
        self.lbl_eta.setText("No dataset loaded")
        self.dataset_cleared.emit()
        self.update_buttons_state()

# -------------------- Dataset Viewer Tab (Lazy Loading) -------------------- #
class DatasetViewerTab(QWidget):
    """
    Displays a responsive table of image thumbnails.
    Initially displays 1/10 of the images; as the user scrolls down,
    additional images (1/10 at a time) are loaded.
    A combo box allows filtering, and a statistics label shows counts per sign.
    """
    def __init__(self):
        super(DatasetViewerTab, self).__init__()
        self.layout = QVBoxLayout(self)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.refresh_table)
        self.layout.addWidget(self.filter_combo)
        self.stats_label = QLabel("Dataset statistics will appear here.")
        self.layout.addWidget(self.stats_label)
        self.tableWidget = QTableWidget()
        self.tableWidget.setSelectionMode(QAbstractItemView.NoSelection)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layout.addWidget(self.tableWidget)

        self.all_data = []    # List of tuples: (image, label)
        self.thumb_size = QSize(100, 100)
        self.page_size = 0    # images per page
        self.current_max_index = 0

        self.tableWidget.verticalScrollBar().valueChanged.connect(self.check_scroll)

    def load_dataset(self, images, labels):
        if images is None or len(images) == 0:
            self.all_data = []
        else:
            self.all_data = list(zip(images, labels))
        unique_labels = sorted(list(set(labels))) if labels is not None else []
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        for label in unique_labels:
            self.filter_combo.addItem(str(label))
        self.filter_combo.blockSignals(False)
        self.update_statistics()
        total = len(self.all_data)
        self.page_size = math.ceil(total / 10) if total > 0 else 0
        self.current_max_index = min(self.page_size, total)
        self.refresh_table()

    def update_statistics(self):
        if not self.all_data:
            self.stats_label.setText("No dataset loaded.")
            return
        stats = {}
        for _, label in self.all_data:
            stats[label] = stats.get(label, 0) + 1
        stats_text = "Dataset Statistics:\n"
        for label, count in stats.items():
            stats_text += f"{label}: {count} images\n"
        self.stats_label.setText(stats_text)

    def refresh_table(self):
        selected_filter = self.filter_combo.currentText()
        if selected_filter == "All":
            filtered_data = self.all_data
        else:
            filtered_data = [(img, lbl) for img, lbl in self.all_data if str(lbl) == selected_filter]
        display_data = filtered_data[:self.current_max_index]
        total = len(display_data)
        if total == 0:
            self.tableWidget.clearContents()
            self.tableWidget.setRowCount(0)
            return
        available_width = self.tableWidget.viewport().width()
        columns = max(1, available_width // (self.thumb_size.width() + 10))
        rows = (total + columns - 1) // columns
        self.tableWidget.setColumnCount(columns)
        self.tableWidget.setRowCount(rows)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setShowGrid(False)
        self.tableWidget.clearContents()

        for index, (img, lbl) in enumerate(display_data):
            row = index // columns
            col = index % columns
            height, width = img.shape
            qimg = QImage((img * 255).astype('uint8').data, width, height, width, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget = QLabel()
            label_widget.setPixmap(pixmap)
            label_widget.setToolTip(str(lbl))
            label_widget.setAlignment(Qt.AlignCenter)
            self.tableWidget.setCellWidget(row, col, label_widget)

        for r in range(rows):
            self.tableWidget.setRowHeight(r, self.thumb_size.height() + 10)

    def check_scroll(self, value):
        scroll_bar = self.tableWidget.verticalScrollBar()
        if value >= scroll_bar.maximum() - 10:
            if self.filter_combo.currentText() == "All":
                total_filtered = len(self.all_data)
            else:
                total_filtered = len([d for d in self.all_data if str(d[1]) == self.filter_combo.currentText()])
            if self.current_max_index < total_filtered:
                self.current_max_index = min(self.current_max_index + self.page_size, total_filtered)
                self.refresh_table()

    def resizeEvent(self, event):
        super(DatasetViewerTab, self).resizeEvent(event)
        self.refresh_table()

# -------------------- Training Tab -------------------- #
class TrainingTab(QWidget):
    """
    The Training tab allows configuration of hyperparameters, model selection, and displays live training progress.
    It embeds a matplotlib plot to show training loss and validation accuracy.
    After training, the model (and metadata) is available for saving.
    """
    training_finished = pyqtSignal(object, dict)  # model and metadata

    def __init__(self):
        super(TrainingTab, self).__init__()
        self.layout = QVBoxLayout(self)
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Training Data Percentage:"))
        self.slider_train = QSlider(Qt.Horizontal)
        self.slider_train.setMinimum(50)
        self.slider_train.setMaximum(100)
        self.slider_train.setValue(80)
        h_layout.addWidget(self.slider_train)
        self.lbl_train_percent = QLabel("80%")
        h_layout.addWidget(self.lbl_train_percent)
        self.slider_train.valueChanged.connect(lambda val: self.lbl_train_percent.setText(f"{val}%"))
        self.layout.addLayout(h_layout)

        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Alexnet", "Lebron", "Resnet"])
        h_layout2.addWidget(self.model_combo)
        self.layout.addLayout(h_layout2)

        h_layout3 = QHBoxLayout()
        h_layout3.addWidget(QLabel("Batch Size:"))
        self.spin_batch = QSpinBox()
        self.spin_batch.setMinimum(1)
        self.spin_batch.setValue(32)
        h_layout3.addWidget(self.spin_batch)
        h_layout3.addWidget(QLabel("Epochs:"))
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setMinimum(1)
        self.spin_epochs.setValue(30)
        h_layout3.addWidget(self.spin_epochs)
        self.layout.addLayout(h_layout3)

        h_layout4 = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_start.clicked.connect(self.start_training)
        h_layout4.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        h_layout4.addWidget(self.btn_stop)
        self.layout.addLayout(h_layout4)

        self.lbl_progress = QLabel("Progress: N/A")
        self.layout.addWidget(self.lbl_progress)
        self.figure, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.training_thread = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_dataset = None

    def set_dataset(self, dataset):
        self.current_dataset = dataset

    def clear_dataset(self):
        self.current_dataset = None
        QMessageBox.information(self, "Dataset Cleared", "The imported dataset has been removed.")

    def start_training(self):
        if self.current_dataset is None:
            QMessageBox.warning(self, "Warning", "No dataset loaded.")
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        train_percentage = self.slider_train.value() / 100.0
        model_choice = self.model_combo.currentText()
        batch_size = self.spin_batch.value()
        num_epochs = self.spin_epochs.value()
        self.training_thread = TrainingThread(model_choice, self.current_dataset, train_percentage, batch_size, num_epochs, self.device)
        self.training_thread.epoch_signal.connect(self.update_training_progress)
        self.training_thread.finished_signal.connect(self.training_finished_slot)
        self.training_thread.error_signal.connect(self.training_error)
        self.training_thread.start()
        self.start_time = time.time()

    @pyqtSlot(int, float, float, float)
    def update_training_progress(self, epoch, loss, val_acc, elapsed):
        self.lbl_progress.setText(f"Epoch {epoch}: Loss={loss:.4f}, Val Acc={val_acc*100:.2f}%, Elapsed={int(elapsed)} sec")
        self.ax[0].clear()
        self.ax[1].clear()
        epochs = list(range(1, epoch + 1))
        if self.training_thread:
            train_losses = self.training_thread.metadata.get("train_loss", [])
            val_accs = self.training_thread.metadata.get("val_acc", [])
        else:
            train_losses = []
            val_accs = []
        self.ax[0].plot(epochs, train_losses, label="Train Loss")
        self.ax[0].legend()
        self.ax[1].plot(epochs, val_accs, label="Val Acc")
        self.ax[1].legend()
        self.canvas.draw()

    @pyqtSlot(dict)
    def training_finished_slot(self, metrics):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        QMessageBox.information(self, "Training", "Training completed.")
        self.training_finished.emit(self.training_thread.model, self.training_thread.metadata)

    @pyqtSlot(str)
    def training_error(self, msg):
        QMessageBox.critical(self, "Training Error", msg)
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def stop_training(self):
        if self.training_thread:
            self.training_thread.stop()
            self.btn_stop.setEnabled(False)
            self.lbl_progress.setText("Training stopped.")

# -------------------- Prediction Tab -------------------- #
class PredictionTab(QWidget):
    """
    The Prediction tab allows the user to load a saved model, select test images, and run predictions.
    It also supports starting a webcam stream (using OpenCV) to capture live frames, take screenshots,
    and predict the sign. When the webcam is stopped, the webcam display is cleared.
    """
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

##################################
# Main Application Window
##################################

class MainWindow(QMainWindow):
    """
    The main window organizes the application into four tabs:
    Dataset Import, Dataset Viewer, Training, and Prediction.
    """
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Sign Language Recognition Tool")
        self.resize(1000, 800)
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.import_tab = DatasetImportTab()
        self.viewer_tab = DatasetViewerTab()
        self.training_tab = TrainingTab()
        self.prediction_tab = PredictionTab()
        self.tabs.addTab(self.import_tab, "Dataset Import")
        self.tabs.addTab(self.viewer_tab, "Dataset Viewer")
        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.prediction_tab, "Prediction")
        self.import_tab.dataset_loaded.connect(self.on_dataset_loaded)
        self.import_tab.dataset_cleared.connect(self.on_dataset_cleared)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_dataset_loaded(self, images, labels, img_shape):
        self.dataset = SignLanguageDataset(images, labels)
        self.viewer_tab.load_dataset(images, labels)
        self.training_tab.set_dataset(self.dataset)
        self.training_tab.input_shape = img_shape

    @pyqtSlot()
    def on_dataset_cleared(self):
        self.dataset = None
        self.viewer_tab.load_dataset(np.array([]), np.array([]))
        self.training_tab.clear_dataset()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
