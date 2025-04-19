import sys
import os
from PyQt5.QtWidgets import QApplication
from frontend.main_window import MainWindow

def main():
    # 1) Create application
    app = QApplication(sys.argv)

    # 2) Load and apply global stylesheet
    qss_path = os.path.join(os.path.dirname(__file__), "style.qss")
    if os.path.exists(qss_path):
        with open(qss_path, "r") as f:
            app.setStyleSheet(f.read())

    # 3) Instantiate and show main window
    window = MainWindow()
    window.show()

    # 4) Run event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
