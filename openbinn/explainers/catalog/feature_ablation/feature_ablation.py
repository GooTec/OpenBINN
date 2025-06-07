import torch
from ...api import BaseExplainer
from captum.attr import FeatureAblation as FA_Captum
from captum.attr import LayerFeatureAblation

class FeatureAblation(BaseExplainer):
    """
    Provides feature ablation attributions.
    Uses a perturbation-based approach to measure the change in output
    when replacing parts of the input/layer with a baseline reference.
    """

    def __init__(self, model, baseline=None, classification_type="multiclass") -> None:
        """
        Args:
            model (torch.nn.Module): Model on which to make predictions.
            baseline (torch.Tensor, optional): Baseline input for attribution calculation.
            classification_type (str): Type of classification model ('multiclass' or 'binary').
        """
        self.baseline = baseline
        self.classification_type = classification_type
        super(FeatureAblation, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):
        """
        Explain an instance prediction using Feature Ablation.
        """
        self.model.eval()
        self.model.zero_grad()

        # Define label if not provided
        if label is None:
            if self.classification_type == "binary":
                label = (self.model(x.float()) > 0.5).long().view(-1)
            else:
                label = self.model(x.float()).argmax(dim=-1)
        label = label.view(-1)

        # Handle baseline: if None, automatically create one matching x; otherwise, check shape.
        if self.baseline is None:
            self.baseline = torch.zeros_like(x, device=x.device)
        elif self.baseline.shape != x.shape:
            raise ValueError(
                f"Baseline shape {self.baseline.shape} does not match input shape {x.shape}."
            )

        # Compute attributions using Feature Ablation
        ablator = FA_Captum(self.model)
        attribution = ablator.attribute(
            x.float(), 
            target=label, 
            layer_baselines=self.baseline
        )

        return attribution

    def get_layer_explanations(self, inputs, label=None):
        """
        Returns layer-specific explanations using Feature Ablation.
        Args:
            inputs (torch.Tensor): Input tensor to the model.
            label (torch.Tensor, optional): Label for which the explanation is calculated.
        Returns:
            explanations (dict): Dictionary containing attributions for each layer.
        """
        target_layer = self.model.print_layer

        explanations = {}
        self.model.eval()
        self.model.zero_grad()

        # Define label if not provided
        label = torch.zeros_like(label)
        label = label.view(-1)
        label = label.long()


        # Iterate through model layers and compute layer-specific attributions.
        for name, layer in self.model.named_modules():
            # Skip non-learnable layers (e.g., activations, dropout, etc.)
            if not hasattr(layer, 'weight') and not hasattr(layer, 'bias'):
                continue
            elif 'intermediate' in name:
                continue
            elif target_layer < 7 and 'network' in name and int(name.split('.')[-2]) >= target_layer:
                print(target_layer, name)
                continue

            # Initialize LayerFeatureAblation for the current layer.
            layer_ablator = LayerFeatureAblation(self.model, layer)
            attribution = layer_ablator.attribute(
                inputs.float(),
                target=label
            )
            explanations[name] = attribution

        return explanations
