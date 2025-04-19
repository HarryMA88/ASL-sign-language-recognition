import numpy as np
import torch
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image
from ml.transforms import get_test_transforms
from frontend.utils.label_map import label_map

class PredictionPopup(QDialog):
    def __init__(self, model, device, img_array, parent=None):
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Prediction Result")
        self.resize(400, 500)
        self.model = model
        self.device = device
        self.img_array = img_array

        self.layout = QVBoxLayout(self)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.img_label)

        self.pred_label = QLabel("Prediction: N/A")
        self.pred_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.pred_label)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        self.run_prediction()

    def run_prediction(self):
        img = self.img_array
        h, w = img.shape
        qimg = QImage((img * 255).astype("uint8").data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)

        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        transform = get_test_transforms()
        tensor = transform(np.array(img_pil)).unsqueeze(0).to(self.device)


        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1).squeeze().cpu().numpy()
            idx = int(out.argmax(1).item())

        self.ax.clear()
        self.ax.bar(np.arange(len(probs)), probs)
        self.ax.set_title("Output Probabilities")
        self.ax.set_xlabel("Class")
        self.ax.set_ylabel("Probability")
        self.canvas.draw()

        label = label_map.get(idx, f"{idx}")
        self.pred_label.setText(f"Prediction: {label}")
