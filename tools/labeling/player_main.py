# Project Curious Network
# This is the main entry point for the player tool

import sys

from PySide2.QtWidgets import QApplication

from player_interface import PlayerInterface

def main():
    app = QApplication(sys.argv)
    interface = PlayerInterface()
    interface.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
