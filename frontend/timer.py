from PyQt5.QtCore import QObject, QTimer, pyqtSignal

class TrainingTimer(QObject):
    tick     = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer   = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_timeout)
        self._elapsed = 0

    def start(self):
        self._elapsed = 0
        self._timer.start()
        self.tick.emit(0)

    def _on_timeout(self):
        self._elapsed += 1
        self.tick.emit(self._elapsed)

    def stop(self):
        if self._timer.isActive():
            self._timer.stop()
            self.finished.emit()
