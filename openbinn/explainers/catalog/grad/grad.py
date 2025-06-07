"""Placeholder for the unused Gradient explainer."""

import torch
from ...api import BaseExplainer


class Gradient(BaseExplainer):
    """Empty implementation for plain gradient explanations."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("Gradient explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("Gradient explainer is not implemented.")
