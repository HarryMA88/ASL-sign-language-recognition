import time
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

class WebcamThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self._is_running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._is_running:
            ret, frame = cap.read()
            if ret:
                self.frame_signal.emit(frame)
            time.sleep(0.03)
        cap.release()

    def stop(self):
        self._is_running = False