"""Placeholder for the unused Feature Ablation explainer."""

import torch
from ...api import BaseExplainer


class FeatureAblation(BaseExplainer):
    """Empty implementation for Feature Ablation explanations."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("FeatureAblation explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("FeatureAblation explainer is not implemented.")
