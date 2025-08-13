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

        # === 안정적인 레이어 선택 방식으로 교체 ===
        core_model = getattr(self.model, "model", self.model)
        network = getattr(core_model, "network", None)
        # target_layer 이전의 레이어만 선택
        layers_to_explain = list(network[:target_layer]) if network is not None else []

        for idx, layer in enumerate(layers_to_explain):
            # Captum의 LayerGradientXActivation을 사용합니다.
            lig = LayerGradientXActivation(self.model, layer)
            
            if self.classification_type == "binary":
                attr = lig.attribute(inputs.float(), target=0)
            else:
                attr = lig.attribute(inputs.float(), target=label)
            
            # 일관성을 위해 인덱스 기반으로 이름을 저장합니다.
            explanations[f"layer_{idx}"] = attr

        return explanations
