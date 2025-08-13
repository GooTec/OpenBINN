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

        # === 안정적인 레이어 선택 방식으로 교체 ===
        core_model = getattr(self.model, "model", self.model)
        network = getattr(core_model, "network", None)
        # target_layer 이전의 레이어만 선택
        layers_to_explain = list(network[:target_layer]) if network is not None else []

        for idx, layer in enumerate(layers_to_explain):
            # 기본 설명자를 먼저 정의하고 NoiseTunnel로 감쌉니다.
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
            
            # 일관성을 위해 인덱스 기반으로 이름을 저장합니다.
            explanations[f"layer_{idx}"] = attr

        return explanations
