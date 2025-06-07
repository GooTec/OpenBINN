"""Placeholder for the unused Integrated Gradients explainer."""

import torch
from ...api import BaseExplainer


class IntegratedGradients(BaseExplainer):
    """Empty implementation for Integrated Gradients."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("IntegratedGradients explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("IntegratedGradients explainer is not implemented.")
