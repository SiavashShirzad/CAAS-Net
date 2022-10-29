import matplotlib.pyplot as plt
import numpy as np


class visualizer:

    def __init__(self, model, dataset):
        self.prediction = model.predict(dataset)
        self.dataset = dataset

    def visualize_all(self):
        for x, y in self.dataset:
            print(x[0]['input_1'].shape)
            plt.figure(figsize=(10, 10))
            ax = plt.subplot(1, 3, 1)
            plt.imshow(x[0][:, :, 0])
            ax = plt.subplot(1, 3, 2)
            plt.imshow(y[0])
            ax = plt.subplot(1, 3, 3)
            plt.imshow(np.argmax(self.prediction[0], axis=-1))
            plt.show()

