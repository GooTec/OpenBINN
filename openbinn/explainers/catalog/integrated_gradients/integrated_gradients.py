import torch
from ...api import BaseExplainer
from captum.attr import IntegratedGradients as IG_Captum
from captum.attr import LayerIntegratedGradients as LayerIG

class IntegratedGradients(BaseExplainer):
    """
    Provides integrated gradient attributions.
    Original paper: https://arxiv.org/abs/1703.01365
    """

    def __init__(self, model, method: str = 'gausslegendre', multiply_by_inputs: bool = False, baseline=None, classification_type="multiclass") -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
            method (str): Integration method to use (e.g., 'gausslegendre')
            multiply_by_inputs (bool): Whether to multiply the attributions by the inputs
            baseline (torch.Tensor, optional): Baseline input for attribution calculation
            classification_type (str): Type of classification model ('multiclass' or 'binary')
        """
        self.method = method
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline
        self.classification_type = classification_type

        super(IntegratedGradients, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):
        """
        Explain an instance prediction.
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

        # Compute attributions using Integrated Gradients
        ig = IG_Captum(self.model, self.multiply_by_inputs)
        attribution = ig.attribute(
            x.float(), target=label, method=self.method, baselines=self.baseline
        )

        return attribution

    def get_layer_explanations(self, inputs, label=None):
        """
        Returns explanations for each layer in the model.
        Args:
            inputs (torch.Tensor): Input tensor to the model.
            label (torch.Tensor, optional): Label for which the explanation is calculated.
        Returns:
            explanations (dict): Dictionary containing explanations for each layer.
        """
        target_layer = self.model.print_layer

        explanations = {}
        self.model.eval()
        self.model.zero_grad()

        # Define label if not provided
        label = torch.zeros_like(label)
        label = label.view(-1)
        label = label.long()

        # Iterate through model layers and compute layer-specific attributions
        for name, layer in self.model.named_modules():
            # Skip non-learnable layers (e.g., activation functions, dropout, etc.)
            if not hasattr(layer, 'weight') and not hasattr(layer, 'bias'):
                continue
            elif 'intermediate' in name:
                continue
            elif target_layer < 7 and 'network' in name and int(name.split('.')[-2]) >= target_layer:
                print(target_layer, name)
                continue

            # Initialize LayerIntegratedGradients for the current layer
            layer_ig = LayerIG(self.model, layer)
            attribution = layer_ig.attribute(
                inputs.float(),
                target=label,
            )
            explanations[name] = attribution

        return explanations
