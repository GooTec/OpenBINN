import torch
from captum.attr import Saliency as Gradient_Captum
from ...api import BaseExplainer


class Gradient(BaseExplainer):
    """
    A baseline approach for computing input attribution.
    It returns the gradients with respect to inputs.
    https://arxiv.org/pdf/1312.6034.pdf
    """

    def __init__(self, model, absolute_value: bool = False) -> None:
        """
        Args:
            model (torch.nn.Module): model on which to make predictions
        """
        self.abs = absolute_value

        super(Gradient, self).__init__(model)

    def get_explanations(self, x: torch.Tensor, label=None) -> torch:
        """
        Explain an instance prediction.
        Args:
            x (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance): feature tensor
            label (torch.Tensor, [N x ...]): labels to explain
            abs (bool): Returns absolute value of gradients if set to True, otherwise returns the (signed) gradients if False
        Returns:
            exp (torch.Tensor, [N x 1 x d] for tabular instance; [N x m x n x d] for image instance: instance level explanation):
        """
        self.model.eval()
        self.model.zero_grad()
        label = self.model(x.float()).argmax(dim=-1) if label is None else label

        saliency = Gradient_Captum(self.model)

        attribution = saliency.attribute(x.float(), target=label, abs=self.abs)

        return attribution

    def get_layer_explanations(self, inputs, label=None):
        '''
        Returns explanations for each layer in the model
        :param inputs: Input tensor to the model
        :param label: Label for which the explanation is calculated (optional)
        :return: Dictionary containing explanations for each layer
        '''
        explanations = {}
        current_input = inputs.clone().detach()

        for name, layer in self.model.named_children():
            if isinstance(layer, torch.nn.ModuleList):
                # Handle ModuleList explicitly
                for i, sub_layer in enumerate(layer):
                    current_output = sub_layer(current_input)

                    # Calculate gradients for the current sub-layer's output
                    saliency = Gradient_Captum(sub_layer)
                    layer_label = current_output.argmax(dim=-1) if label is None else label

                    layer_attribution = saliency.attribute(
                        current_input.float(),
                        target=layer_label,
                        abs=self.abs
                    )

                    # Store the explanation in the dictionary
                    explanations[f"{name}[{i}]"] = torch.FloatTensor(layer_attribution)
                    current_input = current_output
            else:
                # Handle standard layers
                current_output = layer(current_input)

                # Calculate gradients for the current layer's output
                saliency = Gradient_Captum(layer)
                layer_label = current_output.argmax(dim=-1) if label is None else label

                layer_attribution = saliency.attribute(
                    current_input.float(),
                    target=layer_label,
                    abs=self.abs
                )

                # Store the explanation in the dictionary
                explanations[name] = torch.FloatTensor(layer_attribution)
                current_input = current_output

        return explanations
