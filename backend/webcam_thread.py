import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

# This is a background thread for the webcam
class WebcamThread(QThread):
    # Displays the current frame from the webcam
    frame_signal = pyqtSignal(np.ndarray)

    # Constructor to set up the webcam
    def __init__(self):
        super(WebcamThread, self).__init__()
        self._is_running = True
        self.cap = None

    def run(self):
        # Turns on the device's default webcam
        self.cap = cv2.VideoCapture(0)
        # Continuously capture input from the webcam
        while self._is_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_signal.emit(frame)
            time.sleep(0.03)
        # Turns off the webcam
        self.cap.release()

    # Method to stop the webcam
    def stop(self):
        self._is_running = False
