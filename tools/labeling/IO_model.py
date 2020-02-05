import os
import numpy as np


class FileHandler:

    def __init__(self, path):
        if path == '':
            raise IOError('A valid path must be given to instanciate a FileHandler object')
        self.directory = path
        self.frames = os.listdir(path)
        self.frame_counter = 0

    def next_frame(self):
        n_frame = self.frames[self.frame_counter]
        if self.frame_counter < len(self.frames):
            self.frame_counter += 1
        return n_frame

    def previous_frame(self):
        p_frame = self.frames[self.frame_counter]
        if self.frame_counter > 0:
            self.frame_counter -= 1
        return p_frame

    def set_frames_directory(self, path):
        self.directory = path
        self.frames = os.listdir(self.directory)
        self.frame_counter = 0

    def tags_to_file(self, annotated_matrix, path):
        print('Annotated matrix is: {}'.format(annotated_matrix))
        np.savetxt(path, annotated_matrix, delimiter=',', fmt='%f')
