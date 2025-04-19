from PyQt5.QtWidgets import QLabel
from frontend.timer import TrainingTimer

class TimerWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__("00 : 00", parent)
        # Larger, bold timer font
        self.setStyleSheet("font-size: 32pt; font-weight: bold; color: #FFFFFF;")
        self._timer = TrainingTimer(self)
        self._timer.tick.connect(self._update_display)

    def _update_display(self, secs):
        m, s = divmod(secs, 60)
        self.setText(f"{m:02d} : {s:02d}")

    def start(self):
        # Reset display and start timer from 00:00
        self._timer.stop()
        self.setText("00 : 00")
        self._timer.start()

    def stop(self):
        # Stop the timer
        self._timer.stop()
