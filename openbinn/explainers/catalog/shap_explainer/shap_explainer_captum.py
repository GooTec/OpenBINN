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

        # === 안정적인 레이어 선택 방식으로 교체 ===
        core_model = getattr(self.model, "model", self.model)
        network = getattr(core_model, "network", None)
        # target_layer 이전의 레이어만 선택
        layers_to_explain = list(network[:target_layer]) if network is not None else []

        for idx, layer in enumerate(layers_to_explain):
            # Captum의 LayerDeepLiftShap을 사용합니다.
            ldls = LayerDeepLiftShap(self.model, layer, multiply_by_inputs=self.multiply_by_inputs)
            
            if self.classification_type == "binary":
                attr = ldls.attribute(inputs.float(), baselines=baseline, target=0)
            else:
                attr = ldls.attribute(inputs.float(), baselines=baseline, target=label)
            
            # 일관성을 위해 인덱스 기반으로 이름을 저장합니다.
            explanations[f"layer_{idx}"] = attr

        return explanations
