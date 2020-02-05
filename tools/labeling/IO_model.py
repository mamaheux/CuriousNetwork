import os
import glob
import numpy as np

MAX_FRAME_COUNT = 1000000000

class FileHandler:

    def __init__(self, path=os.curdir):
        self._directory = path
        self._frames = glob.glob(os.path.join(path, '*.jpg'))
        self._frame_counter = 0
        self._current_frame = ''
        self._current_annonated_frame = None

    def get_current_frame(self):
        return os.path.join(self._directory, self._current_frame), self._current_annonated_frame

    def next_frame(self):
        if self._frame_counter < len(self._frames)-1:
            self._frame_counter += 1
        else:
            self._frame_counter = 0

        self._current_frame = self._frames[self._frame_counter]
        self._load_annonated_frame()

        return self.get_current_frame()

    def previous_frame(self):
        if self._frame_counter > 0:
            self._frame_counter -= 1
        else:
            self._frame_counter = len(self._frames) - 1

        self._current_frame = self._frames[self._frame_counter]
        self._load_annonated_frame()

        return self.get_current_frame()

    def set_frames_directory(self, path):
        self._directory = path
        self._frames = glob.glob(os.path.join(path, '*.jpg'))
        self._frames.sort(key=self._filename_sort_key)
        self._frame_counter = 0

        self._current_frame = self._frames[self._frame_counter]
        self._load_annonated_frame()

    def tags_to_file(self, annotated_matrix):
        file_name = os.path.join(self._directory, str.replace(self._current_frame, '.jpg', '.txt'))
        np.savetxt(file_name, annotated_matrix, delimiter=',', fmt='%f')

    def _load_annonated_frame(self):
        self._current_annonated_frame = None
        path = os.path.join(self._directory, str.replace(self._current_frame, '.jpg', '.txt'))
        if os.path.exists(path):
            self._current_annonated_frame = np.loadtxt(path, delimiter=',').astype(int)

    def _filename_sort_key(self, x):
        x = str.replace(x, '.jpg', '')
        x = x[(x.rindex(os.path.sep)) + 1:]
        x = x.split('-')

        return MAX_FRAME_COUNT * int(x[0]) + int(x[1])
