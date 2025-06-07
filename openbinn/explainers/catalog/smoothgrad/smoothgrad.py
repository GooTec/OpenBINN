"""Placeholder for the unused SmoothGrad explainer."""

import torch
from ...api import BaseExplainer


class SmoothGrad(BaseExplainer):
    """Empty implementation for SmoothGrad explanations."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("SmoothGrad explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("SmoothGrad explainer is not implemented.")
