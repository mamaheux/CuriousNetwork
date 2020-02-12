import numpy as np

from database.database_loader import DatabaseLoader

class ValidationLoss:
    def __init__(self, database_folder_path, model):
        self._database_loader = DatabaseLoader(database_folder_path)
        self._model = model

    def validate_loss(self):
        J = 0
        for i in range(self._database_loader.get_frames_size()):
            frame = self._database_loader.get_frame(i)
            annotation = self._database_loader.get_frame_annotation(i)

            # error = self._model.predict(frame)
            error = np.zeros(annotation.shape)  # TODO remove

            J += np.sum(annotation * error - (1 - annotation)*error)

        return J / self._database_loader.get_frames_size()
