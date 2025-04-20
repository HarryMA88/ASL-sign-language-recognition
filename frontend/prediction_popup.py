import numpy as np
import torch
from PIL import Image

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from ml.transforms import get_test_transforms
from frontend.utils.label_map import label_map


class PredictionPopup(QDialog):
    """
    Popup dialog showing:
      - Original grayscale image
      - Predicted class label
      - Bar chart of output probabilities
    """

    def __init__(self, model, device, img_array, parent=None):
        super().__init__(parent)

        # Window setup
        self.setWindowTitle("Prediction Result")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(500, 600)

        # Store references
        self.model = model
        self.device = device
        self.img_array = img_array

        # Layout container
        layout = QVBoxLayout(self)

        # Image display
        self.img_label = QLabel(alignment=Qt.AlignCenter)
        layout.addWidget(self.img_label)

        # Prediction text
        self.pred_label = QLabel("Prediction: N/A", alignment=Qt.AlignCenter)
        layout.addWidget(self.pred_label)

        # Probability chart
        self.fig, self.ax = plt.subplots()
        self._style_chart()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Run inference and update UI
        self.run_prediction()

    def _style_chart(self):
        """Apply grey background and white/orange styling to the chart."""
        self.fig.patch.set_facecolor('#2E2E2E')
        self.ax.set_facecolor('#2E2E2E')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        # Set spine colours to white
        for spine in self.ax.spines.values():
            spine.set_color('white')

    def run_prediction(self):
        """
        Render the input image, perform model inference, and plot probabilities.
        """
        # Display the grayscale image
        img = (self.img_array * 255).astype('uint8')
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)

        # Prepare tensor for model
        pil_img = Image.fromarray(img)
        transform = get_test_transforms()
        tensor = transform(np.array(pil_img)).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            out = self.model(tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy().squeeze()
            idx = int(out.argmax(1).item())

        # Plot probabilities
        self.ax.clear()
        self.ax.bar(np.arange(len(probs)), probs, color='orange')
        self.ax.set_title('Output Probabilities', color='white')
        self.ax.set_xlabel('Class', color='white')
        self.ax.set_ylabel('Probability', color='white')
        self.canvas.draw()

        # Update prediction text
        label = label_map.get(idx, str(idx))
        self.pred_label.setText(f"Prediction: {label}")
