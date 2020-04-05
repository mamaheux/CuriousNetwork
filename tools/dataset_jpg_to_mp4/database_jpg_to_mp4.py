import cv2
import numpy as np
import os
from os.path import isfile, join
from IO_model import FileHandler

pathIn= '/home/ggs22/repos/CuriousNetwork/run_output'

file_handle = FileHandler(pathIn)

pathOut = 'video.avi'
fps = 24

size = (1920, 1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'3IVD'), fps, size)
for image_path, image_annotation in file_handle:
    # writing to a image array
    out.write(cv2.imread(image_path))



out.release()