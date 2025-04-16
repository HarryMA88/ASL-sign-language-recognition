import sys
import os
import time
import json
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
    QSlider, QSpinBox, QMessageBox, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
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

# Define three different CNN architectures

class CustomCNN(nn.Module):
    """
    A custom-designed CNN.
    Assumes input images are grayscale and that the image dimensions are provided.
    """
    def __init__(self, input_shape, num_classes):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        # Compute flatten dimension dynamically using a dummy forward pass.
        dummy_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        x = self.pool(nn.functional.relu(self.conv2(nn.functional.relu(self.conv1(dummy_input)))))
        self.flatten_dim = x.numel()
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

class StandardCNN1(nn.Module):
    """
    A standard CNN architecture (variation 1).
    """
    def __init__(self, input_shape, num_classes):
        super(StandardCNN1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        dummy_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        x = self.pool(nn.functional.relu(self.conv1(dummy_input)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        self.flatten_dim = x.numel()
        self.fc1 = nn.Linear(self.flatten_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class StandardCNN2(nn.Module):
    """
    A standard CNN architecture (variation 2).
    """
    def __init__(self, input_shape, num_classes):
        super(StandardCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        dummy_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        x = self.pool(nn.functional.relu(self.conv2(nn.functional.relu(self.conv1(dummy_input)))))
        self.flatten_dim = x.numel()
        self.fc1 = nn.Linear(self.flatten_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.flatten_dim)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

##############################################
# QThread Subclasses for Background Process  #
##############################################

class ImportThread(QThread):
    """
    A thread to import the dataset from a CSV file.
    Emits progress updates (with a rough ETA) and returns images, labels, and image shape.
    Assumes that the CSV file has the label in the first column and flattened pixel values in the remaining columns.
    Rows with labels “J” and “Z” are skipped.
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
            # Process the CSV in chunks
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
                        continue  # skip rows that do not form a square image
                    image = pixels.reshape(img_dim, img_dim)
                    images_list.append(image)
                    labels_list.append(label)
                    total_rows += 1
                    # Update progress (assumes ~10,000 rows total as a rough estimate)
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
        self.dataset = dataset  # an instance of SignLanguageDataset
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
            # Determine input shape from the first sample
            sample_img, _ = self.dataset[0]
            input_shape = sample_img.shape[1:]  # (H, W)
            num_classes = 36  # 26 letters (excluding J and Z) + 10 digits

            # Build the model based on the selected architecture using the new names:
            if self.model_choice == "Alexnet":
                model = CustomCNN(input_shape, num_classes)
            elif self.model_choice == "Lebron":
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
                    # Convert labels (which are characters) into class indices:
                    # Letters: A=0, B=1, …, Z=25; Digits: 0-9 => 26-35.
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

                # Validation phase
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
            # Save metadata and the final model in this thread for later saving
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
            self.model = model  # attach model to thread for access after training
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

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
            time.sleep(0.03)  # roughly 30 fps
        self.cap.release()

    def stop(self):
        self._is_running = False

###########################################
# PyQt GUI: Tab Widgets and Main Window   #
###########################################

class DatasetImportTab(QWidget):
    """
    The Dataset Import tab allows the user to select a CSV file.
    A progress bar shows the import progress and an ETA, and the user can stop the process.
    Once finished, the tab emits the loaded images, labels, and image shape.
    Also provides a button to remove the imported dataset.
    """
    dataset_loaded = pyqtSignal(np.ndarray, np.ndarray, tuple)
    dataset_cleared = pyqtSignal()  # signal emitted when dataset is removed

    def __init__(self):
        super(DatasetImportTab, self).__init__()
        self.layout = QVBoxLayout(self)
        # Import Dataset Button
        self.btn_import = QPushButton("Import Dataset CSV")
        self.btn_import.clicked.connect(self.import_dataset)
        self.layout.addWidget(self.btn_import)
        # Remove Dataset Button
        self.btn_remove = QPushButton("Remove Imported Dataset")
        self.btn_remove.clicked.connect(self.remove_dataset)
        self.layout.addWidget(self.btn_remove)
        # Progress indicators
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)
        self.lbl_eta = QLabel("ETA: N/A")
        self.layout.addWidget(self.lbl_eta)
        self.btn_stop = QPushButton("Stop Import")
        self.btn_stop.clicked.connect(self.stop_import)
        self.btn_stop.setEnabled(False)
        self.layout.addWidget(self.btn_stop)
        self.import_thread = None

    def import_dataset(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Dataset CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return
        self.btn_import.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.import_thread = ImportThread(file_path)
        self.import_thread.progress_signal.connect(self.update_progress)
        self.import_thread.finished_signal.connect(self.import_finished)
        self.import_thread.start()

    @pyqtSlot(int, str)
    def update_progress(self, value, eta):
        self.progress_bar.setValue(value)
        self.lbl_eta.setText(f"ETA: {eta}")

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def import_finished(self, images, labels, img_shape):
        self.btn_import.setEnabled(True)
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
        # Clear progress indicators and emit signal so other tabs can clear the dataset.
        self.progress_bar.setValue(0)
        self.lbl_eta.setText("No dataset loaded")
        self.dataset_cleared.emit()

from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QLabel, QAbstractItemView
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap

class DatasetViewerTab(QWidget):
    """
    The Dataset Viewer tab displays a responsive table of images (as thumbnails) with rows and columns.
    The table automatically adjusts the number of columns based on the available width and allows vertical scrolling.
    A combo box is provided for filtering by sign and a statistics label shows counts per sign.
    """
    def __init__(self):
        super(DatasetViewerTab, self).__init__()
        self.layout = QVBoxLayout(self)
        # Filter combo box for selecting specific sign
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self.populate_table)
        self.layout.addWidget(self.filter_combo)
        # Statistics label
        self.stats_label = QLabel("Dataset statistics will appear here.")
        self.layout.addWidget(self.stats_label)
        # Table widget for image thumbnails
        self.tableWidget = QTableWidget()
        self.tableWidget.setSelectionMode(QAbstractItemView.NoSelection)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.layout.addWidget(self.tableWidget)
        # Stored images and labels
        self.images = None
        self.labels = None
        # Thumbnail size (can be adjusted)
        self.thumb_size = QSize(100, 100)

    def load_dataset(self, images, labels):
        self.images = images
        self.labels = labels
        unique_labels = sorted(list(set(labels))) if labels is not None else []
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        for label in unique_labels:
            self.filter_combo.addItem(str(label))
        self.filter_combo.blockSignals(False)
        self.update_statistics()
        self.populate_table()

    def update_statistics(self):
        if self.labels is None or len(self.labels) == 0:
            self.stats_label.setText("No dataset loaded.")
            return
        stats = {}
        for label in self.labels:
            stats[label] = stats.get(label, 0) + 1
        stats_text = "Dataset Statistics:\n"
        for label, count in stats.items():
            stats_text += f"{label}: {count} images\n"
        self.stats_label.setText(stats_text)

    def populate_table(self):
        # Filter images based on combo box selection
        if self.images is None or self.labels is None:
            self.tableWidget.clearContents()
            self.tableWidget.setRowCount(0)
            return

        selected_filter = self.filter_combo.currentText()
        filtered_data = [(img, lbl) for img, lbl in zip(self.images, self.labels)
                         if selected_filter == "All" or str(lbl) == selected_filter]

        total = len(filtered_data)
        if total == 0:
            self.tableWidget.clearContents()
            self.tableWidget.setRowCount(0)
            return

        # Determine how many columns to display based on the table width and the thumbnail size
        available_width = self.tableWidget.viewport().width()
        columns = max(1, available_width // (self.thumb_size.width() + 10))
        rows = (total + columns - 1) // columns

        self.tableWidget.setColumnCount(columns)
        self.tableWidget.setRowCount(rows)
        # Optional: Hide headers
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.setShowGrid(False)

        # Clear any previous items
        self.tableWidget.clearContents()

        for index, (img, lbl) in enumerate(filtered_data):
            row = index // columns
            col = index % columns
            # Create a QLabel to show the thumbnail image
            height, width = img.shape
            pixmap = QPixmap.fromImage(
                QImage((img * 255).astype('uint8').data, width, height, width, QImage.Format_Grayscale8)
            ).scaled(self.thumb_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label_widget = QLabel()
            label_widget.setPixmap(pixmap)
            label_widget.setToolTip(str(lbl))
            label_widget.setAlignment(Qt.AlignCenter)
            self.tableWidget.setCellWidget(row, col, label_widget)

        # Resize rows to fit content
        for row in range(rows):
            self.tableWidget.setRowHeight(row, self.thumb_size.height() + 10)

    def resizeEvent(self, event):
        """Recalculate the table columns when the widget is resized."""
        super(DatasetViewerTab, self).resizeEvent(event)
        self.populate_table()


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
        # Training data percentage slider
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
        # Model selection (using new names)
        h_layout2 = QHBoxLayout()
        h_layout2.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Alexnet", "Lebron", "Resnet"])
        h_layout2.addWidget(self.model_combo)
        self.layout.addLayout(h_layout2)
        # Hyperparameters: batch size and epochs
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
        # Start and Stop buttons
        h_layout4 = QHBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_start.clicked.connect(self.start_training)
        h_layout4.addWidget(self.btn_start)
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        h_layout4.addWidget(self.btn_stop)
        self.layout.addLayout(h_layout4)
        # Live training progress label
        self.lbl_progress = QLabel("Progress: N/A")
        self.layout.addWidget(self.lbl_progress)
        # Live training plot (matplotlib)
        self.figure, self.ax = plt.subplots(1, 2, figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.training_thread = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_dataset = None  # will be set from main window

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
        # Update live plot
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

class PredictionTab(QWidget):
    """
    The Prediction tab allows the user to load a saved model, select test images, and run predictions.
    It also supports starting a webcam stream (using OpenCV) to capture live frames and predict the sign.
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
        self.webcam_label = QLabel()
        self.layout.addWidget(self.webcam_label)
        self.webcam_thread = None
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
            # Map the new model names to the appropriate classes.
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
            # Convert prediction index to label: 0-25 -> A-Z, 26-35 -> 0-9.
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

    @pyqtSlot(np.ndarray)
    def update_webcam_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(320, 240, Qt.KeepAspectRatio)
        self.webcam_label.setPixmap(pixmap)

##################################
# Main Application Window        #
##################################

class MainWindow(QMainWindow):
    """
    The main window organizes the application into four tabs:
    - Dataset Import
    - Dataset Viewer
    - Training
    - Prediction
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
        # Connect the dataset import completion to update viewer and training tabs.
        self.import_tab.dataset_loaded.connect(self.on_dataset_loaded)
        self.import_tab.dataset_cleared.connect(self.on_dataset_cleared)

    @pyqtSlot(np.ndarray, np.ndarray, tuple)
    def on_dataset_loaded(self, images, labels, img_shape):
        self.dataset = SignLanguageDataset(images, labels)
        self.viewer_tab.load_dataset(images, labels)
        self.training_tab.set_dataset(self.dataset)
        # Save the input shape in training metadata for model reconstruction.
        self.training_tab.input_shape = img_shape

    @pyqtSlot()
    def on_dataset_cleared(self):
        # Clear the dataset in viewer and training tabs.
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
