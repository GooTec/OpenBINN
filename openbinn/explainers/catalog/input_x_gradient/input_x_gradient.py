import torch
from ...api import BaseExplainer
from captum.attr import InputXGradient as InputXGradient_Captum
from captum.attr import LayerGradientXActivation as LayerGXA


class InputTimesGradient(BaseExplainer):
    """
    A baseline approach for computing the attribution.
    It multiplies input with the gradient with respect to input.
    https://arxiv.org/abs/1605.01713
    """

    def __init__(self, model):
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        super(InputTimesGradient, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None):
        """
        Explain an instance prediction.
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """
        self.model.eval()
        self.model.zero_grad()
        label = self.model(x.float()).argmax(dim=-1) if label is None else label

        input_x_gradient = InputXGradient_Captum(self.model)

        attribution = input_x_gradient.attribute(x.float(), target=label)

        return attribution

    def get_layer_explanations(self, inputs, label=None):
        """
        Returns explanations for each layer in the model using Layer Gradient X Activation.
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

            # Initialize LayerGradientXActivation for the current layer
            layer_gxa = LayerGXA(self.model, layer)

            # Compute attributions for the current layer
            attribution = layer_gxa.attribute(
                inputs.float(),
                target=label,
                attribute_to_layer_input=False  # Set to True if needed
            )

            # Store the explanation for the current layer
            explanations[name] = attribution

        return explanations