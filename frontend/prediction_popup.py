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
    """
    A popup dialog to show the result of a single prediction.

    It shows:
      - the original image
      - the predicted label
      - a bar chart of output probabilities
    """
    def __init__(self, model, device, img_array, parent=None):
        super().__init__(parent)
        # Remove the help button on Windows
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setWindowTitle("Prediction Result")
        self.resize(500, 600)

        self.model = model
        self.device = device
        self.img_array = img_array

        # Main vertical layout
        layout = QVBoxLayout(self)

        # Label to show the input image
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.img_label)

        # Label to show the predicted class text
        self.pred_label = QLabel("Prediction: N/A")
        self.pred_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pred_label)

        # Matplotlib canvas for probability bars
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Run the model and update UI
        self.run_prediction()

    def run_prediction(self):
        """
        Display the image, run the model, then plot and show results.
        """
        # Show the scaled grayscale image
        img = self.img_array
        h, w = img.shape
        qimg = QImage((img * 255).astype("uint8").data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(
            200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.img_label.setPixmap(pix)

        # Transform to tensor via PIL then torchvision pipeline
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        transform = get_test_transforms()
        tensor = transform(np.array(img_pil)).unsqueeze(0).to(self.device)

        # Predict without tracking gradients
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.nn.functional.softmax(out, dim=1).squeeze().cpu().numpy()
            idx = int(out.argmax(1).item())

        # Clear and draw probability bar chart
        self.ax.clear()
        self.ax.bar(np.arange(len(probs)), probs)
        self.ax.set_title("Output Probabilities")
        self.ax.set_xlabel("Class")
        self.ax.set_ylabel("Probability")
        self.canvas.draw()

        # Map index to label and display it
        label = label_map.get(idx, f"{idx}")
        self.pred_label.setText(f"Prediction: {label}")
