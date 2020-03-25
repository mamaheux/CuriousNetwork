import time

from curious_dataset import CuriousDataset

class ModelExecutionTime:
    def __init__(self, dataset_folder_path, dataset_normalization, model):
        self._dataset = CuriousDataset(dataset_folder_path, normalization=dataset_normalization)
        self._model = model

    def calculate(self):
        forward_elapsed_time = 0
        backward_elapsed_time = 0

        for i in range(len(self._dataset)):
            image, _ = self._dataset[i]

            start = time.time()
            output = self._model(image.unsqueeze(0))
            forward_elapsed_time += time.time() - start

            start = time.time()
            loss = output.sum()
            loss.backward()
            backward_elapsed_time += time.time() - start

        forward_elapsed_time /= len(self._dataset)
        backward_elapsed_time /= len(self._dataset)

        return forward_elapsed_time, backward_elapsed_time
