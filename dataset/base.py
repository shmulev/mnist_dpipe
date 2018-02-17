from abc import abstractmethod, ABC, ABCMeta

import numpy as np


class Dataset(ABC):
    @property
    @abstractmethod
    def ids(self):
        """Returns a tuple of ids of all objects in the dataset."""


class ClassificationDataset(Dataset):
    """Abstract class that describes a dataset for medical image classification."""

    @abstractmethod
    def load_image(self, identifier: str) -> np.array:
        """
        Loads a dataset entry given its identifier
        Parameters
        ----------
        identifier: str
            object's identifier
        Returns
        -------
        object:
            The entry corresponding to the identifier
        """

    @abstractmethod
    def load_label(self, identifier: str) -> int:
        """
        Loads a label for a given identifier
        Parameters
        ----------
        identifier: str
            object's identifier
        Returns
        -------
        int:
            The label corresponding to the identifier
        """

    @property
    @abstractmethod
    def n_chans_image(self) -> int:
        """
        The number of channels in the input image's tensor

        Returns
        -------
        channels: int
        """