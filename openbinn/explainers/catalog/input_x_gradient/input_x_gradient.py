"""Input×Gradient explainer wrapping Captum."""

import torch
from ...api import BaseExplainer
from captum.attr import InputXGradient as CaptumInputXGradient
from captum.attr import LayerGradientXActivation


class InputTimesGradient(BaseExplainer):
    """Simple gradient multiplied by the input."""

    def __init__(self, model, classification_type: str = "multiclass") -> None:
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

        ig = CaptumInputXGradient(self.model)

        if self.classification_type == "binary":
            attr = ig.attribute(x.float(), target=0)
        else:
            attr = ig.attribute(x.float(), target=label)

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
            if target_layer < 7 and "network" in name:
                parts = name.split(".")
                if len(parts) > 1 and parts[-2].isdigit() and int(parts[-2]) >= target_layer:
                    continue

            lig = LayerGradientXActivation(self.model, layer)
            if self.classification_type == "binary":
                attr = lig.attribute(inputs.float(), target=0, allow_unused=True)
            else:
                attr = lig.attribute(inputs.float(), target=label, allow_unused=True)
            explanations[name] = attr

        return explanations
