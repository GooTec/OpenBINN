"""Gradient SHAP explainer wrapping Captum."""

import torch
from ...api import BaseExplainer
from captum.attr import GradientShap as CaptumGradientShap
from captum.attr import LayerGradientShap


class GradientShap(BaseExplainer):
    """Attributions using randomized gradients."""

    def __init__(
        self,
        model,
        baseline=None,
        n_samples: int = 50,
        stdevs: float = 0.0,
        classification_type: str = "multiclass",
    ) -> None:
        self.baseline = baseline
        self.n_samples = n_samples
        self.stdevs = stdevs
        self.classification_type = classification_type
        super().__init__(model)

    def _infer_label(self, x: torch.Tensor, label):
        if label is None:
            if self.classification_type == "binary":
                label = (self.model(x.float()) > 0.5).long().view(-1)
            else:
                label = self.model(x.float()).argmax(dim=-1)
        return label.view(-1)

    def _get_baseline(self, x: torch.Tensor):
        if self.baseline is None:
            return torch.zeros_like(x, device=x.device)
        if self.baseline.shape != x.shape:
            raise ValueError(
                f"Baseline shape {self.baseline.shape} does not match input shape {x.shape}."
            )
        return self.baseline

    def get_explanations(self, x: torch.Tensor, label=None):
        self.model.eval()
        self.model.zero_grad()

        label = self._infer_label(x, label)
        baseline = self._get_baseline(x)

        gs = CaptumGradientShap(self.model)

        if self.classification_type == "binary":
            attr = gs.attribute(
                x.float(), baselines=baseline, target=0, n_samples=self.n_samples, stdevs=self.stdevs
            )
        else:
            attr = gs.attribute(
                x.float(), baselines=baseline, target=label, n_samples=self.n_samples, stdevs=self.stdevs
            )

        return attr

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):
        target_layer = getattr(self.model, "print_layer", 0)

        self.model.eval()
        self.model.zero_grad()

        label = self._infer_label(inputs, label)
        baseline = self._get_baseline(inputs)

        explanations = {}
        for name, layer in self.model.named_modules():
            if not hasattr(layer, "weight") and not hasattr(layer, "bias"):
                continue
            if "intermediate" in name:
                continue
            if target_layer < 7 and "network" in name:
                parts = name.split(".")
                if len(parts) > 1:
                    try:
                        if int(parts[-2]) >= target_layer:
                            continue
                    except ValueError:
                        pass

            lgs = LayerGradientShap(self.model, layer)
            if self.classification_type == "binary":
                attr = lgs.attribute(
                    inputs.float(), baselines=baseline, target=0, n_samples=self.n_samples, stdevs=self.stdevs
                )
            else:
                attr = lgs.attribute(
                    inputs.float(), baselines=baseline, target=label, n_samples=self.n_samples, stdevs=self.stdevs
                )
            explanations[name] = attr

        return explanations
