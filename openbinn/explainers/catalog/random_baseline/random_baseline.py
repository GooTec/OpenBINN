"""Placeholder for the unused random baseline explainer."""

import torch
from ...api import BaseExplainer


class RandomBaseline(BaseExplainer):
    """Empty implementation for random baseline explanations."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("RandomBaseline explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("RandomBaseline explainer is not implemented.")
