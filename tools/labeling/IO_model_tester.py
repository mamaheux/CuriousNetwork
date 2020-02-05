from IO_model import FileHandler
import numpy as np
import os

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
# tags = np.zeros(9, dtype=int)

print(tags.shape)
print(type(tags))
print(type(tags) == 'class \'numpy.ndarray\'')

model.tags_to_file(tags, './test.csv')