import cv2
from IO_model import FileHandler
import numpy as np

pathIn= '/home/ggs22/repos/CuriousNetwork/run_output'

file_handle = FileHandler(pathIn)

pathOut = 'video.avi'
fps = 24

size = (1920, 1080)
out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'3IVD'), fps, size)
alpha = 0.85



for image_path, image_annotation in file_handle:

    # writing to a image array
    img = cv2.imread(image_path)

    annotations = np.where(image_annotation == 1)
    for index in range(annotations[0].shape[0]):
        x = annotations[1][index] * 120
        y = annotations[0][index] * 120
        layer = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillConvexPoly(layer, np.array([[x, y], [x + 120, y], [x + 120, y + 120], [x, y + 120]]),
                           (0, 0, 255), 8, 0)
        img = cv2.addWeighted(layer, alpha, img, 1, 0.0)

    out.write(img)



out.release()