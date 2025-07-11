"""SmoothGrad explainer built on gradient × input."""

import torch
from ...api import BaseExplainer
from captum.attr import NoiseTunnel
from captum.attr import InputXGradient as CaptumInputXGradient
from captum.attr import LayerGradientXActivation


class SmoothGrad(BaseExplainer):
    """Adds noise and averages gradient×input attributions."""

    def __init__(
        self,
        model,
        n_samples: int = 20,
        stdevs: float = 0.1,
        classification_type: str = "multiclass",
    ) -> None:
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

    def get_explanations(self, x: torch.Tensor, label=None):
        self.model.eval()
        self.model.zero_grad()

        label = self._infer_label(x, label)

        base = CaptumInputXGradient(self.model)
        nt = NoiseTunnel(base)

        if self.classification_type == "binary":
            attr = nt.attribute(
                x.float(),
                nt_type="smoothgrad",
                stdevs=self.stdevs,
                n_samples=self.n_samples,
                target=0,
            )
        else:
            attr = nt.attribute(
                x.float(),
                nt_type="smoothgrad",
                stdevs=self.stdevs,
                n_samples=self.n_samples,
                target=label,
            )

        return attr

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):
        target_layer = getattr(self.model, "print_layer", 0)

        self.model.eval()
        self.model.zero_grad()

        label = self._infer_label(inputs, label)

        explanations = {}
        for name, layer in self.model.named_modules():
            if not hasattr(layer, "weight") and not hasattr(layer, "bias"):
                continue
            if "intermediate" in name:
                continue
            if target_layer < 7 and "network" in name and int(name.split(".")[-2]) >= target_layer:
                continue

            base = LayerGradientXActivation(self.model, layer)
            nt = NoiseTunnel(base)
            if self.classification_type == "binary":
                attr = nt.attribute(
                    inputs.float(),
                    nt_type="smoothgrad",
                    stdevs=self.stdevs,
                    n_samples=self.n_samples,
                    target=0,
                )
            else:
                attr = nt.attribute(
                    inputs.float(),
                    nt_type="smoothgrad",
                    stdevs=self.stdevs,
                    n_samples=self.n_samples,
                    target=label,
                )
            explanations[name] = attr

        return explanations
