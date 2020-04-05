import os
import glob
import numpy as np

MAX_FRAME_COUNT = 1000000000

class FileHandler:

    def __init__(self, path=os.curdir):
        self._directory = path
        self._frames = glob.glob(os.path.join(path, '*.jpg'))
        self._frames.sort(key=self._filename_sort_key)

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        return os.path.join(self._directory, self._frames[i]), self._load_annonated_frame(i)

    def _load_annonated_frame(self, i):
        path = os.path.join(self._directory, str.replace(self._frames[i], '.jpg', '.txt'))
        if os.path.exists(path):
            return np.loadtxt(path, delimiter=',').astype(int)
        else:
            return np.zeros(shape=(16, 9), dtype=int)

    def _filename_sort_key(self, x):
        x = str.replace(x, '.jpg', '')
        x = x[(x.rindex(os.path.sep)) + 1:]
        x = x.split('-')

        return MAX_FRAME_COUNT * int(x[0]) + int(x[1])
