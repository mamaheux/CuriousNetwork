import numpy as np

from metrics.roc_curve import RocCurve

class ValidationLoss:
    def __init__(self, database_folder_path, model, roc_curve_thresholds):
        self._roc_curve = RocCurve(database_folder_path, model, roc_curve_thresholds)

    def calculate(self):
        rates = self._roc_curve.calculate()
        return np.trapz(rates[1, :], x=rates[0, :])
