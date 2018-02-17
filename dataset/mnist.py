from .base import ClassificationDataset

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MNIST(ClassificationDataset):
    def __init__(self, data_path: str):
        # fetch mnist datasets
        mnist = input_data.read_data_sets(data_path, reshape=False, one_hot=False)
        X_train, y_train = mnist.train.images, mnist.train.labels
        X_validation, y_validation = mnist.validation.images, mnist.validation.labels
        X_test, y_test = mnist.test.images, mnist.test.labels

        # transope the matrices so `channels` is the secord index
        X_train = np.transpose(X_train, [0, 3, 1, 2])
        X_validation = np.transpose(X_validation, [0, 3, 1, 2])
        X_test = np.transpose(X_test, [0, 3, 1, 2])

        self._images = np.concatenate((X_train, X_validation, X_test), axis=0)
        self._labels = np.concatenate((y_train, y_validation, y_test), axis=0)
        self._n_chans_image = X_train.shape[1]
        self._ids = list(np.arange(self._labels.shape[0]).astype(str))

    def load_image(self, identifier: str) -> np.array:
        return self._images[int(identifier), ...]

    def load_label(self, identifier: str) -> int:
        return self._labels[int(identifier)]

    @property
    def ids(self) -> list:
        return self._ids

    @property
    def n_chans_image(self) -> int:
        return self._n_chans_image
