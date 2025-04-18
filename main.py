import sys
import os
from PyQt5.QtWidgets import QApplication
from frontend.main_window import MainWindow

def main():
    app = QApplication(sys.argv)
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
