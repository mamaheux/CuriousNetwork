import os
import glob
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from skimage import io

MAX_FRAME_COUNT = 1000000000

class CuriousDataset(Dataset):
    def __init__(self, folder_path):
        self._folder_path = folder_path
        self._images = glob.glob(os.path.join(self._folder_path, '*.jpg'))
        self._images.sort(key=self._filename_sort_key)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        return self.get_image(index)

    def get_image(self, index):
        int8_image = io.imread(os.path.join(self._folder_path, self._images[index]))
        float_image = int8_image.astype(np.float32)
        float_image = np.moveaxis(float_image, -1, 0)
        min = np.min(float_image)
        max = np.max(float_image)
        return torch.from_numpy((float_image - min) / (max - min))

    def get_frame_annotation(self, index):
        self._current_annonated_frame = None
        path = os.path.join(self._folder_path, str.replace(self._images[index], '.jpg', '.txt'))
        if os.path.exists(path):
            return np.loadtxt(path, delimiter=',').astype(int)
        else:
            return None

    def _filename_sort_key(self, x):
        x = str.replace(x, '.jpg', '')
        x = x[(x.rindex(os.path.sep)) + 1:]
        x = x.split('-')

        return MAX_FRAME_COUNT * int(x[0]) + int(x[1])

def create_dataset_loader(folder_path, batch_size):
    return DataLoader(CuriousDataset(folder_path), batch_size=batch_size, shuffle=True, num_workers=4)
