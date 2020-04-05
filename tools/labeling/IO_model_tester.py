# Project Curious Network
# This scripts implements a short test routine for the annotation tool

from IO_model import FileHandler
import numpy as np

print('No Robustness: it will open any files, not just images (*.jpg, *.png, etc')
model = FileHandler()
print(model.next_frame())
print(model.next_frame())

# TODO This value is hard coded for a local machine, it must be adapted
model = FileHandler('/home/ggs22/Pictures')

print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())
print(model.next_frame())

print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())
print(model.previous_frame())

tags = np.zeros((9,16), dtype=float)

print(tags.shape)
print(type(tags))
print(type(tags) == 'class \'numpy.ndarray\'')

model.tags_to_file(tags, './test.csv')
