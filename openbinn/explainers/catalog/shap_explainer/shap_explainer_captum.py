"""Placeholder for the unused DeepLiftShap explainer."""

import torch
from ...api import BaseExplainer


class DeepLiftShapExplainer(BaseExplainer):
    """Empty implementation for DeepLiftShap explanations."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("DeepLiftShap explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("DeepLiftShap explainer is not implemented.")
