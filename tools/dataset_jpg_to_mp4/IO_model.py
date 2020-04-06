import os
import glob
import numpy as np
import cv2

MAX_FRAME_COUNT = 1000000000

class FileHandler:

    def __init__(self, path_curiosity, path_test):
        self._directory_curiosity = path_curiosity
        self._directory_test = path_test

        self._frames_curiosity = glob.glob(os.path.join(path_curiosity, '*.jpg'))
        self._frames_curiosity.sort(key=self._filename_sort_key)

        self._frames_test = glob.glob(os.path.join(path_test, '*.jpg'))
        self._frames_test.sort(key=self._filename_sort_key)

    def __len__(self):
        return len(self._frames_curiosity)

    def __getitem__(self, i):
        return os.path.join(self._directory_curiosity, self._frames_curiosity[i]),\
               self._load_annonated_frame(i), \
               os.path.join(self._directory_test, self._frames_test[i]), \
               self._load_test_frame(i)

    def get_annotated_pair(self, path):
        annotated_frames = glob.glob(os.path.join(path, 'annoted_img_*.jpg'))
        annotated_frames.sort()
        test_frames = glob.glob(os.path.join(path, 'test_img_*.jpg'))
        test_frames.sort()
        assert len(annotated_frames) == len(test_frames)

        return test_frames, annotated_frames

    def _load_annonated_frame(self, i):
        path = os.path.join(self._directory_curiosity, str.replace(self._frames_curiosity[i], '.jpg', '.txt'))
        if os.path.exists(path):
            return np.loadtxt(path, delimiter=',').astype(int)
        else:
            return np.zeros(shape=(16, 9), dtype=int)

    def _load_test_frame(self, i):
        path = os.path.join(self._directory_test, str.replace(self._frames_test[i], '.jpg', '.txt'))
        if os.path.exists(path):
            return np.loadtxt(path, delimiter=',').astype(int)
        else:
            return np.zeros(shape=(16, 9), dtype=int)

    def _filename_sort_key(self, x):
        x = str.replace(x, '.jpg', '')
        x = x[(x.rindex(os.path.sep)) + 1:]
        x = x.split('-')

        return MAX_FRAME_COUNT * int(x[0]) + int(x[1])

    def generate_annotated_frames_pairs(self, output_path, alpha=0.7):
        delta = 120

        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        for index_global, (curiosity_img_path, curiosity_annotation, test_img_path, test_annotation) in enumerate(self):

            filename_curiosity = os.path.join(output_path, f'annoted_img_{index_global:05d}.jpg')
            filename_test = os.path.join(output_path, f'test_img_{index_global:05d}.jpg')

            img_curiosity = cv2.imread(curiosity_img_path)
            img_test = cv2.imread(test_img_path)

            annotations = np.where(test_annotation == 1)
            for index in range(annotations[0].shape[0]):
                x = annotations[1][index] * delta
                y = annotations[0][index] * delta
                layer = np.zeros(img_test.shape, dtype=np.uint8)
                cv2.fillConvexPoly(layer, np.array([[x, y], [x + delta, y], [x + delta, y + delta], [x, y + delta]]),
                                   (0, 0, 255), 8, 0)
                img_test = cv2.addWeighted(layer, alpha, img_test, 1, 0.0)

            annotations = np.where(curiosity_annotation == 1)
            for index in range(annotations[0].shape[0]):
                x = annotations[1][index] * delta
                y = annotations[0][index] * delta
                layer = np.zeros(img_curiosity.shape, dtype=np.uint8)
                cv2.fillConvexPoly(layer, np.array([[x, y], [x + delta, y], [x + delta, y + delta], [x, y + delta]]),
                                   (0, 0, 255), 8, 0)
                img_curiosity = cv2.addWeighted(layer, alpha, img_curiosity, 1, 0.0)

            print(f'dual (Curiosity & test) annotation overlay: {index_global+1}/{len(self)}')
            assert cv2.imwrite(filename_curiosity, img_curiosity)
            assert cv2.imwrite(filename_test, img_test)
