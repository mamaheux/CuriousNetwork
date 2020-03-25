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
            image, annotation = self._dataset[i]

            positive_count += np.sum(annotation == 1)
            negative_count += np.sum(annotation == 0)

            error = self._model(image.unsqueeze(0)).detach().numpy()
            error = np.sqrt(error)

            for threshold in self._thresholds:
                predicted_roi = error > threshold
                true_positive_counts[threshold] += np.sum(annotation * predicted_roi)
                false_positive_counts[threshold] += np.sum((np.ones(annotation.shape) - annotation) * predicted_roi)

        rates = np.zeros((3, len(self._thresholds)))
        rates[2, :] = self._thresholds

        for i in range(len(self._thresholds)):
            rates[1, i] = true_positive_counts[self._thresholds[i]] / positive_count
            rates[0, i] = false_positive_counts[self._thresholds[i]] / negative_count

        sorted_indexes = np.argsort(rates[0, :])
        for i in range(rates.shape[0]):
            rates[i, :] = rates[i, sorted_indexes]

        return rates

    def save(self, output_path):
        rates = self.calculate()
        np.savetxt(output_path + '.txt', rates, delimiter=',', fmt='%f')


        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111)

        ax.plot(rates[0, :],rates[1, :], label='rates')
        ax.plot(np.array([0, 1]), np.array([0, 1]),'--', label='rates')
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')

        fig.savefig(output_path + '.png')
