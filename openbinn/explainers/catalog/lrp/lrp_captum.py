import torch
from ...api import BaseExplainer
from captum.attr import LRP as LRP_Captum
from captum.attr import LayerLRP as LayerLRP

class LRP(BaseExplainer):
    """
    Provides Layer-wise Relevance Propagation (LRP) attributions.
    """

    def __init__(self, model) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        super(LRP, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):
        """
        Explain an instance prediction using LRP.
        Args:
            x (torch.Tensor): Input tensor to the model
            label (torch.Tensor): Target label for which the explanation is calculated (optional)
        Returns:
            torch.Tensor: LRP attributions
        """
        self.model.eval()
        self.model.zero_grad()

        # Define label if not provided
        label = self.model(x.float()).argmax(dim=-1) if label is None else label
        label = label.view(-1)  # Flatten label if necessary

        # Compute attributions using LRP
        lrp = LRP_Captum(self.model)
        attribution = lrp.attribute(
            x.float(), target=label
        )

        return attribution

    def get_layer_explanations(self, inputs, label=None):
        """
        Returns explanations for each layer in the model using Layer LRP.
        Args:
            inputs (torch.Tensor): Input tensor to the model
            label (torch.Tensor): Label for which the explanation is calculated (optional)
        Returns:
            explanations (dict): Dictionary containing explanations for each layer
        """
        explanations = {}
        self.model.eval()
        self.model.zero_grad()

        # Define label if not provided
        if label is None:
            label = self.model(inputs.float()).argmax(dim=-1)

        # Iterate through the model layers
        for name, layer in self.model.named_modules():
            # Skip non-learnable layers (like nn.ReLU, nn.Dropout, etc.)
            if not hasattr(layer, 'weight') and not hasattr(layer, 'bias'):
                continue

            # Initialize LayerLRP for the current layer
            layer_lrp = LayerLRP(self.model, layer)

            # Compute attributions for the current layer
            attribution = layer_lrp.attribute(
                inputs.float(),
                target=label,
                attribute_to_layer_input=False  # Set to True if needed
            )

            # Store the explanation for the current layer
            explanations[name] = attribution

        return explanations
