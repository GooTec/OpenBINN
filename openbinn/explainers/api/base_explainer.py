import torch
from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    """
    Abstract class to implement custom explanation methods for a given.
    Parameters
    ----------
    mlmodel: xai-bench.models.MLModel
        Classifier we wish to explain.
    Methods
    -------
    get_explanations:
        Generate explanations for given input.
    Returns
    -------
    None
    """

    def __init__(self, mlmodel):
        self.model = mlmodel

    @abstractmethod
    def get_explanations(self, inputs: torch.Tensor, label: torch.Tensor):
        """
        Generate explanations for given input/s.
        Parameters
        ----------
        inputs: torch.tensor
            Input in two-dimensional shape (m, n).
        label: torch.tensor
            Label
        Returns
        -------
        torch.tensor
            Explanation vector/matrix.
        """
        pass

    @abstractmethod
    def get_layer_explanations(self, inputs:torch.Tensor, label: torch.Tensor): 
        """
        Generate explanations for intermediate layers.
        Parameters
        ----------
        inputs: torch.Tensor
            Input in two-dimensional shape (m, n).
        Returns
        -------
        dict
            Dictionary where keys are layer names and values are the explanations (importance scores).
        """
        pass

