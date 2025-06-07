import numpy as np
import torch
from ...api import BaseExplainer
from .pytorch_lrp.innvestigator import InnvestigateModel
from openbinn.experiment_utils import convert_to_numpy


class LRP(BaseExplainer):
    """
    Provides layer-wise relevance propagation explanations with respect to the input layer.
    Uses the InnvestigateModel for explanation generation.
    """

    def __init__(self, model, lrp_exponent=2, method="b-rule", beta=1) -> None:
        """
        Args:
            model (torch.nn.Module): model to generate explanations for
            lrp_exponent (float): LRP exponent (default: 2)
            method (str): LRP method to use, e.g., "e-rule" (default: "e-rule")
            beta (float): Beta parameter for LRP, if applicable (default: 0.5)
        """
        super(LRP, self).__init__(model)
        self.model = model
        self.innvestigate_model = InnvestigateModel(
            model, lrp_exponent=lrp_exponent, method=method, beta=beta
        )

    def get_explanations(self, x: torch.FloatTensor, label=None) -> torch.FloatTensor:
        """
        Generate LRP explanations for the input data.

        Args:
            x (torch.FloatTensor): Input data [N x d]
            label (torch.LongTensor or None): Labels to explain [N]. If None, predictions are used.

        Returns:
            torch.FloatTensor: Explanation attributions [N x d]
        """
        self.model.eval()

        # Predict labels if not provided
        if label is None:
            label = self.model(x.float()).argmax(dim=-1)
        else:
            label = convert_to_numpy(label)

        # Ensure input is in the correct shape
        x = x.float()

        # Generate explanations
        attribution_scores = []
        for i in range(x.shape[0]):  # Iterate over samples in the batch
            _, relevance = self.innvestigate_model.innvestigate(
                in_tensor=x[i : i + 1], rel_for_class=int(label[i])
            )
            attribution_scores.append(relevance.squeeze(0).detach().numpy())

        # Convert explanations to a tensor
        return torch.FloatTensor(attribution_scores)
    
    def get_layer_explanations(self, x: torch.FloatTensor, label=None):
        """
        Generate layer-wise relevance explanations for the input data.

        Args:
            x (torch.FloatTensor): Input data [N x d]
            label (torch.LongTensor or None): Labels to explain [N]. If None, predictions are used.

        Returns:
            dict: A dictionary where keys are layer names (e.g., 'Layer_1') and values are
                  relevance attributions for that layer.
        """
        self.model.eval()

        # Predict labels if not provided
        if label is None:
            label = self.model(x.float()).argmax(dim=-1)
        else:
            label = convert_to_numpy(label)

        # Ensure input is in the correct shape
        x = x.float()

        # Generate layer-wise explanations
        layer_explanations = {}
        for i in range(x.shape[0]):  # Iterate over samples in the batch
            _, layer_relevances = self.innvestigate_model.investigate_layer(
                in_tensor=x[i : i + 1], rel_for_class=int(label[i])
            )
            for layer_name, relevance in layer_relevances.items():
                if layer_name not in layer_explanations:
                    layer_explanations[layer_name] = []
                layer_explanations[layer_name].append(relevance.numpy())

        # Aggregate layer-wise explanations
        for layer_name in layer_explanations:
            layer_explanations[layer_name] = np.stack(layer_explanations[layer_name])

        return layer_explanations
