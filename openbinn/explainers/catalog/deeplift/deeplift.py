import torch
from ...api import BaseExplainer
from captum.attr import DeepLift as DeepLift_Captum
from captum.attr import LayerDeepLift as LayerDeepLift

class DeepLift(BaseExplainer):
    """
    Provides DeepLift attributions.
    Original paper: https://arxiv.org/abs/1704.02685
    """

    def __init__(self, model, multiply_by_inputs: bool = False, baseline=None, classification_type="multiclass") -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
            multiply_by_inputs (bool): Whether to multiply the attributions by the inputs
            baseline (torch.Tensor, optional): Baseline input for attribution calculation
            classification_type (str): Type of classification model ('multiclass' or 'binary')
        """
        self.multiply_by_inputs = multiply_by_inputs
        self.baseline = baseline
        self.classification_type = classification_type

        super(DeepLift, self).__init__(model)

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
            else:  # multiclass
                label = self.model(x.float()).argmax(dim=-1)

        label = label.view(-1)  # Flatten label if necessary

        # Handle baseline shape mismatch
        if self.baseline is None:
            # Automatically create a baseline matching the shape of x
            self.baseline = torch.zeros_like(x, device=x.device)
        elif self.baseline.shape != x.shape:
            raise ValueError(
                f"Baseline shape {self.baseline.shape} does not match input shape {x.shape}."
            )

        # Compute attributions
        dl = DeepLift_Captum(self.model, self.multiply_by_inputs)

        if self.classification_type == "binary":
            # Use single output node for binary classification
            attribution = dl.attribute(x.float(), target=0, baselines=self.baseline)
        else:
            # Use specified label for multiclass classification
            attribution = dl.attribute(x.float(), target=label, baselines=self.baseline)

        return attribution

    def get_layer_explanations(self, inputs, label=None):
        """
        Returns explanations for each layer in the model.
        Args:
            inputs (torch.Tensor): Input tensor to the model
            label (torch.Tensor): Label for which the explanation is calculated (optional)
        Returns:
            explanations (dict): Dictionary containing explanations for each layer
        """
        target_layer = self.model.print_layer
        # print(target_layer)

        explanations = {}
        self.model.eval()
        self.model.zero_grad()

        # Define label if not provided
        if label is None:
            if self.classification_type == "binary":
                label = (self.model(inputs.float()) > 0.5).long().view(-1)
            else:
                label = self.model(inputs.float()).argmax(dim=-1)

        # Iterate through the model layers
        for name, layer in self.model.named_modules():
            # Skip non-learnable layers (like nn.ReLU, nn.Dropout, etc.)
            if not hasattr(layer, 'weight') and not hasattr(layer, 'bias'):
                continue
            elif 'intermediate' in name:
                continue
            elif target_layer < 7 and 'network' in name and int(name.split('.')[-2]) >= target_layer:
                # print(target_layer, name)
                continue

            # Initialize LayerDeepLift for the current layer
            layer_dl = LayerDeepLift(self.model, layer)

            # Compute attributions for the current layer
            if self.classification_type == "binary":
                attribution = layer_dl.attribute(
                    inputs.float(),
                    target=0,
                    baselines=self.baseline,
                    attribute_to_layer_input=False  # Set to True if needed
                )
            else:
                attribution = layer_dl.attribute(
                    inputs.float(),
                    target=label,
                    baselines=self.baseline,
                    attribute_to_layer_input=False  # Set to True if needed
                )

            # Store the explanation for the current layer
            explanations[name] = attribution

        return explanations
