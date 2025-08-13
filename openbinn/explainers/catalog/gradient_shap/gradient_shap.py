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

        # === 안정적인 레이어 선택 방식으로 교체 ===
        core_model = getattr(self.model, "model", self.model)
        network = getattr(core_model, "network", None)
        # target_layer 이전의 레이어만 선택
        layers_to_explain = list(network[:target_layer]) if network is not None else []

        for idx, layer in enumerate(layers_to_explain):
            # LayerGradientShap을 사용합니다.
            lgs = LayerGradientShap(self.model, layer)
            
            if self.classification_type == "binary":
                attr = lgs.attribute(
                    inputs.float(),
                    baselines=baseline,
                    target=0,
                    n_samples=self.n_samples,
                    stdevs=self.stdevs
                )
            else:
                attr = lgs.attribute(
                    inputs.float(),
                    baselines=baseline,
                    target=label,
                    n_samples=self.n_samples,
                    stdevs=self.stdevs
                )
            # 일관성을 위해 인덱스 기반으로 이름을 저장합니다.
            explanations[f"layer_{idx}"] = attr

        return explanations
