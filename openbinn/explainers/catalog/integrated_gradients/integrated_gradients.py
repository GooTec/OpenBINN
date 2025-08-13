"""Integrated Gradients explainer wrapping Captum."""

import torch
from ...api import BaseExplainer
from captum.attr import IntegratedGradients as CaptumIntegratedGradients
from captum.attr import LayerIntegratedGradients


class IntegratedGradients(BaseExplainer):
    """Provides Integrated Gradients attributions."""

    def __init__(
        self,
        model,
        multiply_by_inputs: bool = True,
        baseline=None,
        n_steps: int = 50,
        classification_type: str = "multiclass",
    ) -> None:
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline
        self.n_steps = n_steps
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

        ig = CaptumIntegratedGradients(self.model, multiply_by_inputs=self.multiply_by_inputs)

        if self.classification_type == "binary":
            attribution = ig.attribute(
                x.float(), baselines=baseline, target=0, n_steps=self.n_steps
            )
        else:
            attribution = ig.attribute(
                x.float(), baselines=baseline, target=label, n_steps=self.n_steps
            )

        return attribution

    def get_layer_explanations(self, inputs: torch.Tensor, label=None):
        target_layer = getattr(self.model, "print_layer", 0)
        self.model.eval()
        self.model.zero_grad()
        label = self._infer_label(inputs, label)
        baseline = self._get_baseline(inputs)
        explanations = {}

        # === DeepLIFT에서 가져온 안정적인 레이어 선택 방식으로 교체 ===
        core_model = getattr(self.model, "model", self.model)
        network = getattr(core_model, "network", None)
        # target_layer 이전의 레이어만 선택
        layers_to_explain = list(network[:target_layer]) if network is not None else []

        for idx, layer in enumerate(layers_to_explain):
            lig = LayerIntegratedGradients(self.model, layer)
            
            if self.classification_type == "binary":
                attr = lig.attribute(
                    inputs.float(),
                    baselines=baseline,
                    target=0,
                    n_steps=self.n_steps,
                )
            else:
                attr = lig.attribute(
                    inputs.float(),
                    baselines=baseline,
                    target=label,
                    n_steps=self.n_steps,
                )
            # 이름을 인덱스 기반으로 저장하여 일관성 유지
            explanations[f"layer_{idx}"] = attr

        return explanations
