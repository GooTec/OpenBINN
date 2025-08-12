"""DeepLiftShap explainer wrapping Captum."""

import torch
from ...api import BaseExplainer
from captum.attr import DeepLiftShap as CaptumDeepLiftShap
from captum.attr import LayerDeepLiftShap


class DeepLiftShapExplainer(BaseExplainer):
    """SHAP variant of DeepLift."""

    def __init__(
        self,
        model,
        multiply_by_inputs: bool = False,
        baseline=None,
        classification_type: str = "multiclass",
    ) -> None:
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline
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

        dl_shap = CaptumDeepLiftShap(self.model, multiply_by_inputs=self.multiply_by_inputs)

        if self.classification_type == "binary":
            attr = dl_shap.attribute(x.float(), baselines=baseline, target=0)
        else:
            attr = dl_shap.attribute(x.float(), baselines=baseline, target=label)

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

            ldls = LayerDeepLiftShap(self.model, layer, multiply_by_inputs=self.multiply_by_inputs)
            if self.classification_type == "binary":
                attr = ldls.attribute(inputs.float(), baselines=baseline, target=0)
            else:
                attr = ldls.attribute(inputs.float(), baselines=baseline, target=label)
            explanations[name] = attr

        return explanations
