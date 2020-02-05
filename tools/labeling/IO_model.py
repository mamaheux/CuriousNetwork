import os
import glob
import numpy as np


class FileHandler:

    def __init__(self, path=os.curdir):
        self.directory = path
        self.frames = glob.glob(os.path.join(path, '*.jpg'))
        self.frame_counter = 0
        self.current_frame = ''
        self.current_annonated_frame = ''

    def next_frame(self):
        self.current_frame = self.frames[self.frame_counter]
        self.current_annonated_frame = glob.glob(os.path.join(self.current_frame, '*.txt'))
        if self.frame_counter < len(self.frames)-1:
            self.frame_counter += 1
        else:
            self.frame_counter = 0
        return os.path.join(self.directory, self.current_frame), self.current_annonated_frame

    def previous_frame(self):
        self.current_frame = self.frames[self.frame_counter]
        self.current_annonated_frame = glob.glob(os.path.join(self.current_frame, '*.txt'))
        if self.frame_counter > 0:
            self.frame_counter -= 1
        else:
            self.frame_counter = len(self.frames) - 1
        return os.path.join(self.directory, self.current_frame), self.current_annonated_frame

    def set_frames_directory(self, path):
        self.directory = path
        self.frames = glob.glob(os.path.join(path, '*.jpg'))
        self.frames.sort(key=str.lower)
        self.frame_counter = 0

    def tags_to_file(self, annotated_matrix):
        file_name = os.path.join(self.directory, str.replace(self.current_frame, '.jpg', '.txt'))
        np.savetxt(file_name, annotated_matrix, delimiter=',', fmt='%f')
