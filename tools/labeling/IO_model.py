import os
import numpy as np


class FileHandler:

    def __init__(self, path=os.curdir):
        self.directory = path
        self.frames = os.listdir(path)
        self.frame_counter = 0

    def next_frame(self):
        n_frame = self.frames[self.frame_counter]
        if self.frame_counter < len(self.frames)-1:
            self.frame_counter += 1
        else:
            self.frame_counter = 0
        return n_frame

    def previous_frame(self):
        p_frame = self.frames[self.frame_counter]
        if self.frame_counter > 0:
            self.frame_counter -= 1
        else:
            self.frame_counter = len(self.frames) - 1
        return p_frame

    def set_frames_directory(self, path):
        self.directory = path
        self.frames = os.listdir(self.directory)
        self.frames.sort(key=str.lower)
        print('Frames order:\n{}'.format(self.frames))
        self.frame_counter = 0

    def tags_to_file(self, annotated_matrix, path):
        print('Annotated matrix is: {}'.format(annotated_matrix))
        np.savetxt(path, annotated_matrix, delimiter=',', fmt='%f')
