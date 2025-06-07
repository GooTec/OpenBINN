"""Placeholder for the unused EBP explainer."""

import torch
from ...api import BaseExplainer


class EBP(BaseExplainer):
    """Empty implementation for Excitation Backpropagation."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("EBP explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("EBP explainer is not implemented.")
