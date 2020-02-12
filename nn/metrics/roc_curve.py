import numpy as np
import matplotlib.pyplot as plt

from curious_dataset import CuriousDataset

class RocCurve:
    def __init__(self, database_folder_path, model, thresholds):
        self._dataset = CuriousDataset(database_folder_path)
        self._model = model
        self._thresholds = thresholds

    def calculate(self):
        positive_count = 0
        negative_count = 0

        true_positive_counts = {}
        false_positive_counts = {}

        for threshold in self._thresholds:
            true_positive_counts[threshold] = 0
            false_positive_counts[threshold] = 0

        for i in range(len(self._dataset)):
            image = self._dataset.get_image(i)
            annotation = self._dataset.get_frame_annotation(i)

            positive_count += np.sum(annotation == 1)
            negative_count += np.sum(annotation == 0)

            error = self._model.forward(image).numpy()

            for threshold in self._thresholds:
                predicted_roi = error > threshold
                true_positive_counts[threshold] += np.sum(annotation * predicted_roi)
                false_positive_counts[threshold] += np.sum((np.ones(annotation.shape) - annotation) * predicted_roi)

        rates = np.zeros((2, len(self._thresholds)))
        for i in range(len(self._thresholds)):
            rates[1, i] = true_positive_counts[self._thresholds[i]] / positive_count
            rates[0, i] = false_positive_counts[self._thresholds[i]] / negative_count

        sorted_indexes = np.argsort(rates[0, :])
        rates[0, :] = rates[0, sorted_indexes]
        rates[1, :] = rates[1, sorted_indexes]

        return rates

    def plot(self):
        rates = self.calculate()
        plt.plot(rates[0, :],rates[0, :], label='rates')
        plt.plot(np.array([0, 1]), np.array([0, 1]),'--', label='rates')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.legend()
        plt.show()
