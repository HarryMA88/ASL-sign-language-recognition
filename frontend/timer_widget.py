from PyQt5.QtWidgets import QLabel
from frontend.timer import TrainingTimer

class TimerWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__("00:00", parent)
        # Remove default margins and padding
        self.setContentsMargins(0, 0, 0, 0)
        # Larger, bold timer font, tightly spaced
        self.setStyleSheet(
            "font-size: 28pt; font-weight: bold; color: #FFFFFF;"
            "margin: 0; padding: 0;"
        )

        # Underlying QTimer logic
        self._timer = TrainingTimer(self)
        self._timer.tick.connect(self._update_display)

    def _update_display(self, secs):
        m, s = divmod(secs, 60)
        # Display without extra spaces
        self.setText(f"{m:02d}:{s:02d}")

    def start(self):
        # Reset internal counter and display
        self._timer.stop()
        self._timer._elapsed = 0
        self.setText("00:00")
        self._timer.start()

    def stop(self):
        # Stop counting
        self._timer.stop()