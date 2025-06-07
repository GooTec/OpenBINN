"""Placeholder for the unused LIME explainer."""

import torch
from ...api import BaseExplainer


class LIME(BaseExplainer):
    """Empty implementation for LIME explanations."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("LIME explainer is not implemented.")

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):  # pragma: no cover
        raise NotImplementedError("LIME explainer is not implemented.")
