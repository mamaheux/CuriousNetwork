from PySide2.QtCore import Qt
from PySide2.QtGui import QKeySequence
from PySide2.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout

class LabelingInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CuriousNetwork - Labeling Interface")
        self._create_ui()

    def _create_ui(self):
        self._path_line_edit = QLineEdit()
        self._path_line_edit.setReadOnly(True)

        self._browse_button = QPushButton("Browse")
        self._browse_button.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_B))
        self._browse_button.setMaximumWidth(150)
        self._browse_button.clicked.connect(self._on_browse_button_clicked)

        browse_layout = QHBoxLayout()
        browse_layout.addWidget(self._path_line_edit)
        browse_layout.addWidget(self._browse_button)

        self._previous_button = QPushButton("Previous")
        self._previous_button.setMaximumWidth(150)
        self._previous_button.setShortcut(QKeySequence(Qt.Key_A))
        self._previous_button.clicked.connect(self._on_previous_button_clicked)

        self._next_button = QPushButton("Next")
        self._next_button.setMaximumWidth(150)
        self._next_button.setShortcut(QKeySequence(Qt.Key_D))
        self._next_button.clicked.connect(self._on_next_button_clicked)

        self._clear_button = QPushButton("Clear")
        self._clear_button.setMaximumWidth(150)
        self._clear_button.setShortcut(QKeySequence(Qt.Key_Delete))
        self._clear_button.clicked.connect(self._on_clear_button_clicked)

        self._save_button = QPushButton("Save")
        self._save_button.setMaximumWidth(150)
        self._save_button.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_S))
        self._save_button.clicked.connect(self._on_save_button_clicked)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self._previous_button)
        action_layout.addWidget(self._next_button)
        action_layout.addSpacing(100)
        action_layout.addWidget(self._clear_button)
        action_layout.addSpacing(100)
        action_layout.addWidget(self._save_button)

        layout = QVBoxLayout()
        layout.addLayout(browse_layout)
        layout.addLayout(action_layout)
        self.setLayout(layout)

    def _on_browse_button_clicked(self):
        print("browse")

    def _on_previous_button_clicked(self):
        print("previous")

    def _on_next_button_clicked(self):
        print("next")

    def _on_clear_button_clicked(self):
        print("clear")

    def _on_save_button_clicked(self):
        print("save")