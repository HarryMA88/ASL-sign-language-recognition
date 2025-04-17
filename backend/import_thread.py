import time
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

class ImportThread(QThread):
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(np.ndarray, np.ndarray, tuple)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self._is_running = True

    def run(self):
        start = time.time()
        df_iter = pd.read_csv(self.file_path, chunksize=1000)
        images, labels, total = [], [], 0
        for chunk in df_iter:
            if not self._is_running: break
            for _, row in chunk.iterrows():
                if not self._is_running: break
                lbl = row.iloc[0]
                if lbl in ['J','Z']: continue
                pix = row.iloc[1:].values.astype(np.float32) / 255.0
                dim = int(np.sqrt(len(pix)))
                if dim*dim != len(pix): continue
                images.append(pix.reshape(dim,dim))
                labels.append(lbl)
                total += 1
                elapsed = time.time() - start
                progress = int((total/10000)*100)
                remain  = int((elapsed/total)*(10000-total)) if total else 0
                eta     = f"{remain//60}m {remain%60}s"
                self.progress_signal.emit(min(progress,100), eta)
        if self._is_running and images:
            arr_i = np.array(images)
            arr_l = np.array(labels)
            shape = arr_i[0].shape
            self.finished_signal.emit(arr_i, arr_l, shape)

    def stop(self):
        self._is_running = False
