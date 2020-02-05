import sys

from PySide2.QtWidgets import QApplication

from labeling_interface import LabelingInterface

def main():
    app = QApplication(sys.argv)
    interface = LabelingInterface()
    interface.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()