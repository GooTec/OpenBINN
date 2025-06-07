"""Placeholder for the unused LRP explainer."""

import torch
from ...api import BaseExplainer


class LRP(BaseExplainer):
    """Empty implementation for Layer-wise Relevance Propagation."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("LRP explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("LRP explainer is not implemented.")
