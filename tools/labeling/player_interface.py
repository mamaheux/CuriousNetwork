from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QKeySequence
from PySide2.QtWidgets import QWidget, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel
from IO_model import FileHandler
import os

from labeling_widget import LabelingWidget

FPS = 24

class PlayerInterface(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('CuriousNetwork - Player Interface')
        self._create_ui()

        self._file_handler = FileHandler()

        self._player_mode = ''
        self._is_player_enabled = False

        self._frame_timer = QTimer()
        self._frame_timer.timeout.connect(self._on_frame_timer_timeout)
        self._frame_timer.start(int(1000 / FPS))

    def _create_ui(self):
        self._path_line_edit = QLineEdit()
        self._path_line_edit.setReadOnly(True)

        self._browse_button = QPushButton('Browse')
        self._browse_button.setShortcut(QKeySequence(Qt.CTRL + Qt.Key_B))
        self._browse_button.setMaximumWidth(150)
        self._browse_button.clicked.connect(self._on_browse_button_clicked)

        browse_layout = QHBoxLayout()
        browse_layout.addWidget(self._path_line_edit)
        browse_layout.addWidget(self._browse_button)

        self._file_label = QLabel('')
        self._labeling_widget = LabelingWidget()

        self._rewind_button = QPushButton('Rewind')
        self._rewind_button.setMaximumWidth(150)
        self._rewind_button.pressed.connect(self._on_rewind_button_pressed)
        self._rewind_button.released.connect(self._on_rewind_button_released)

        self._play_button = QPushButton('Play')
        self._play_button.setMaximumWidth(150)
        self._play_button.pressed.connect(self._on_play_button_pressed)
        self._play_button.released.connect(self._on_play_button_released)

        action_layout = QHBoxLayout()
        action_layout.addWidget(self._rewind_button)
        action_layout.addWidget(self._play_button)

        layout = QVBoxLayout()
        layout.addLayout(browse_layout)
        layout.addWidget(self._file_label)
        layout.addWidget(self._labeling_widget)
        layout.addLayout(action_layout)
        self.setLayout(layout)

    def _on_browse_button_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        directory = dialog.getExistingDirectory(self, 'Choose Frames Directory', os.curdir)
        self._file_handler.set_frames_directory(directory)
        self._path_line_edit.setText(directory)

        path, tags = self._file_handler.get_current_frame()
        self._update_image(path, tags)

    def _on_rewind_button_pressed(self):
        print('a')
        self._player_mode = 'rewind'
        self._is_player_enabled = True

    def _on_rewind_button_released(self):
        print('b')
        self._is_player_enabled = False

    def _on_play_button_pressed(self):
        print('c')
        self._player_mode = 'play'
        self._is_player_enabled = True

    def _on_play_button_released(self):
        print('d')
        self._is_player_enabled = False

    def _on_frame_timer_timeout(self):
        if not self._is_player_enabled:
            return

        if self._player_mode == 'rewind':
            path, tags = self._file_handler.previous_frame()
            self._update_image(path, tags)
        elif self._player_mode == 'play':
            path, tags = self._file_handler.next_frame()
            self._update_image(path, tags)

    def _update_image(self, path, tags):
        self._labeling_widget.set_image(path)
        if tags is not None:
            self._labeling_widget.set_tags(tags)

        self._file_label.setText(path)
