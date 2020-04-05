import matplotlib.pyplot as plt

class LearningCurves:
    def __init__(self):
        self._training_loss_values = []
        self._validation_loss_values = []

    def clear(self):
        self._training_loss_values = []
        self._validation_loss_values = []

    def add_training_loss_value(self, value):
        self._training_loss_values.append(value)

    def add_validation_loss_value(self, value):
        self._validation_loss_values.append(value)

    def save_figure(self, output_path):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        epochs = range(1, len(self._training_loss_values) + 1)
        ax1.plot(epochs, self._training_loss_values, '-o')
        ax1.set_title(u'Entraînement')
        ax1.set_xlabel(u'Époque')
        ax1.set_ylabel(u'Coût')

        epochs = range(1, len(self._validation_loss_values) + 1)
        ax2.plot(epochs, self._validation_loss_values, '-o', color='tab:orange')
        ax2.set_title(u'Validation')
        ax2.set_xlabel(u'Époque')
        ax2.set_ylabel(u'Coût')

        fig.savefig(output_path)
