import numpy as np

from curious_dataset import CuriousDataset

class ValidationLoss:
    def __init__(self, database_folder_path, model):
        self._dataset = CuriousDataset(database_folder_path)
        self._model = model

    def calculate(self):
        J = 0
        for i in range(len(self._dataset)):
            image = self._dataset.get_image(i)
            annotation = self._dataset.get_frame_annotation(i)

            error = self._model.forward(image).numpy()
            J += np.sum(annotation * error - (1 - annotation)*error)

        return J / len(self._dataset)
