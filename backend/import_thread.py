import time
import numpy as np
import pandas as pd
from PyQt5.QtCore import QThread, pyqtSignal

class ImportThread(QThread):
    """
    This class is for a background thread to import the dataset so the gui doesnt freeze
    """
    # This is for displaying the progress and eta for importing the dataset
    progress_signal = pyqtSignal(int, str)
    # This is for displaying an indicator for when the dataset is finished importing
    finished_signal = pyqtSignal(np.ndarray, np.ndarray, tuple)

    def __init__(self, file_path):
        """
        This is the constructor and stores the file path of the dataset to be loaded
        """
        super().__init__()
        self.file_path = file_path
        # This is a boolean for indicating that we are currently importing the dataset
        self._is_running = True

    def run(self):
        """
        This is where the thread main logic is in
        """
        start = time.time()

        # This reads the csv file in chunks of 1000 which is better for larger files
        df_iter = pd.read_csv(self.file_path, chunksize=1000)
        images, labels, total = [], [], 0
        # This processes the csv file chunk by chunk
        for chunk in df_iter:
            if not self._is_running: break
            # Iterates through each row in the csv where each row is for each image
            for _, row in chunk.iterrows():
                # Retrieves the image by processing the data in the row
                if not self._is_running: break
                lbl = row.iloc[0]
                if lbl in ['J','Z']: continue
                pix = row.iloc[1:].values.astype(np.float32) / 255.0
                dim = int(np.sqrt(len(pix)))
                if dim*dim != len(pix): continue
                # Stores the images and its labels
                images.append(pix.reshape(dim,dim))
                labels.append(lbl)
                # Keeps track of the progess of the import and calculates the eta and displays them
                total += 1
                elapsed = time.time() - start
                progress = int((total/10000)*100)
                remain = int((elapsed/total)*(10000-total)) if total else 0
                eta = f"{remain//60}m {remain%60}s"
                self.progress_signal.emit(min(progress,100), eta)
        # Displays the finished signal on the gui when done
        if self._is_running and images:
            arr_i = np.array(images)
            arr_l = np.array(labels)
            shape = arr_i[0].shape
            self.finished_signal.emit(arr_i, arr_l, shape)

    def stop(self):
        """
        This method is to stop importing the data
        """
        self._is_running = False
