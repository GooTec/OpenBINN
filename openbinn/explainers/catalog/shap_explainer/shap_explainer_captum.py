import torch
from ...api import BaseExplainer
from captum.attr import DeepLiftShap, LayerDeepLiftShap

class DeepLiftShapExplainer(BaseExplainer):
    """
    Provides DeepLiftShap attributions.
    Uses DeepLiftShap for input-level explanations and LayerDeepLiftShap for layer-wise explanations.
    Based on:
    https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf
    """

    def __init__(self, model, multiply_by_inputs: bool = True, baseline=None, classification_type="binary") -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions.
            multiply_by_inputs (bool): Whether to multiply the attributions by the inputs.
            baseline (torch.Tensor, optional): Baseline input for attribution calculation.
            classification_type (str): 'multiclass' or 'binary'.
        """
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline
        self.classification_type = classification_type
        super(DeepLiftShapExplainer, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):
        """
        Compute DeepLiftShap explanations for the given input.
        Args:
            x (torch.Tensor): Input tensor.
            label (torch.Tensor, optional): Target label for which attributions are computed.
        Returns:
            torch.Tensor: Attributions with the same shape as x.
        """
        self.model.eval()
        self.model.zero_grad()

        # Compute label if not provided, then flatten and cast to long.
        if label is None:
            if self.classification_type == "binary":
                label = (self.model(x.float()) > 0.5).long().view(-1)
            else:
                label = self.model(x.float()).argmax(dim=-1)
        label = label.view(-1).long()

        # Handle baseline: create one matching x if not provided.
        if self.baseline is None:
            self.baseline = torch.zeros_like(x, device=x.device)
        elif self.baseline.shape != x.shape:
            raise ValueError(
                f"Baseline shape {self.baseline.shape} does not match input shape {x.shape}."
            )

        # Compute attributions using DeepLiftShap.
        dls = DeepLiftShap(self.model, multiply_by_inputs=self.multiply_by_inputs)
        attributions = dls.attribute(x.float(), baselines=self.baseline, target=label)
        return attributions

    def get_layer_explanations(self, inputs, label=None):
        """
        Compute layer-wise DeepLiftShap explanations.
        Args:
            inputs (torch.Tensor): Input tensor to the model.
            label (torch.Tensor, optional): Target label for which attributions are computed.
        Returns:
            dict: Dictionary mapping layer names to their respective attributions.
        """
        explanations = {}
        self.model.eval()
        self.model.zero_grad()
        inputs = inputs.clone().detach().requires_grad_(True)

        # 이진 분류이며 모델 출력이 단일 스칼라인 경우 target은 항상 0으로 설정
        if self.classification_type == "binary":
            label = torch.zeros(inputs.shape[0], device=inputs.device).long()
        else:
            if label is None:
                label = self.model(inputs.float()).argmax(dim=-1)
            else:
                label = label.view(-1)
            label = label.long()

        # Iterate through model layers.
        for name, layer in self.model.named_modules():
            # Skip non-learnable layers (예: activation, dropout 등)
            if not hasattr(layer, "weight") and not hasattr(layer, "bias"):
                continue

            # Initialize LayerDeepLiftShap for the current layer.
            layer_dls = LayerDeepLiftShap(self.model, layer, multiply_by_inputs=self.multiply_by_inputs)
            # Compute attributions for the current layer.
            # attribute_to_layer_input: True면 레이어 입력에 대한 기여도를, False면 출력에 대한 기여도를 계산합니다.
            try:
                attribution = layer_dls.attribute(
                    inputs.float(),
                    baselines=self.baseline,
                    target=label,
                    attribute_to_layer_input=False
                )
            except RuntimeError as e:
                if "not have been used in the graph" in str(e):
                    # 해당 레이어에서 미분 경로가 없으면, 0 텐서를 반환하도록 함.
                    attribution = torch.zeros_like(inputs)
                    print(f"Warning: Attribution for layer {name} not computed due to unused tensor error. Returning zeros.")
                else:
                    raise e
            explanations[name] = attribution

        return explanations
