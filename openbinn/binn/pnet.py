import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, ReLU
import pandas as pd

from .base_net import BaseNet
from .util import scatter_nd

class FeatureLayer(torch.nn.Module):
    """This layer will take our input data of size `(N_genes, N_features)`, and perform
    elementwise multiplication of the features of each gene. This is effectively
    collapsing the `N_features dimension`, outputting a single scalar latent variable
    for each gene.
    """

    def __init__(self, num_genes: int, num_features: int):
        super().__init__()
        self.num_genes = num_genes
        self.num_features = num_features
        weights = torch.Tensor(self.num_genes, self.num_features)
        self.weights = nn.Parameter(weights)
        self.bias = nn.Parameter(torch.Tensor(self.num_genes))
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = x * self.weights
        x = torch.sum(x, dim=-1)
        x = x + self.bias
        return x


class SparseLayer(torch.nn.Module):
    """Sparsely connected layer, with connections taken from pnet."""

    def __init__(self, layer_map):
        super().__init__()
        if type(layer_map)==pd.core.frame.DataFrame:
            map_numpy = layer_map.to_numpy()
        else:
            map_numpy=layer_map
        self.register_buffer(
            "nonzero_indices", torch.LongTensor(np.array(np.nonzero(map_numpy)).T)
        )
        self.layer_map = layer_map
        self.shape = map_numpy.shape
        self.weights = nn.Parameter(torch.Tensor(self.nonzero_indices.shape[0], 1))
        self.bias = nn.Parameter(torch.Tensor(self.shape[1]))
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        sparse_tensor = scatter_nd(
            self.nonzero_indices, self.weights.squeeze(), self.shape
        )
        x = torch.mm(x, sparse_tensor)
        # no bias yet
        x = x + self.bias
        return x


class PNet(BaseNet):
    """Implementation of the pnet sparse feedforward network in torch. Uses the same
    pytorch geometric dataset as the message passing networks.
    """

    def __init__(
        self,
        layers,
        num_genes: int,
        num_features: int = 3,
        lr: float = 0.001,
        intermediate_outputs: bool = True,
        class_weights: bool = True,
        scheduler: str = "lambda",
        diversity_lambda: float = 0.0,
        loss_cfg: dict | None = None,
        norm_type: str = "batchnorm",
        dropout_rate: float = 0.1,
        input_dropout: float = 0.5,
        optim_cfg: dict | None = None,
    ):
        """Initialize.
        :param layers: list of pandas dataframes describing the pnet masks for each
            layer
        :param num_genes: number of genes in dataset
        :param num_features: number of features for each gene
        :param lr: learning rate
        :param diversity_lambda: weight for layer diversity penalty
        """
        super().__init__(lr=lr, scheduler=scheduler)
        self.class_weights = class_weights
        self.layers = layers
        self.num_genes = num_genes
        self.num_features = num_features
        self.intermediate_outputs = intermediate_outputs
        self.diversity_lambda = diversity_lambda
        self.optim_cfg = optim_cfg or {}

        self.loss_cfg = {"main": 1.0, "aux": 0.0, "per_layer": None}
        if loss_cfg is not None:
            self.loss_cfg.update(loss_cfg)
        if self.loss_cfg["per_layer"] is None:
            self.loss_cfg["per_layer"] = [1.0] * (len(layers) - 1)

        self.norm_type = norm_type
        self.dropout_rate = dropout_rate
        self.input_dropout = input_dropout

        self.network = nn.ModuleList()
        self.intermediate_outs = nn.ModuleList()

        def _norm(n):
            if self.norm_type == "batchnorm":
                return nn.BatchNorm1d(n)
            if self.norm_type == "layernorm":
                return nn.LayerNorm(n)
            return nn.Identity()

        self.network.append(
            nn.Sequential(
                FeatureLayer(self.num_genes, self.num_features),
                _norm(self.num_genes),
                nn.Tanh(),
            )
        )
        for i, layer_map in enumerate(layers):
            if i != (len(layers) - 1):
                dropout = self.input_dropout if i == 0 else self.dropout_rate
                self.network.append(
                    nn.Sequential(
                        nn.Dropout(p=dropout),
                        SparseLayer(layer_map),
                        _norm(layer_map.shape[1]),
                        nn.Tanh(),
                    )
                )
                if self.intermediate_outputs:
                    self.intermediate_outs.append(nn.Linear(layer_map.shape[0], 1))
            else:
                self.network.append(nn.Linear(layer_map.shape[0], 1))

    def forward(self, x):
        """ Forward pass, output a list containing predictions from each
            intermediate layer, which can be weighted differently during
            training & validation """

        y = []
        x = self.network[0](x)
        for aa in range(1, len(self.network) - 1):
            if self.intermediate_outputs:
                y.append(self.intermediate_outs[aa - 1](x))
            x = self.network[aa](x)
        y.append(self.network[-1](x))

        return y

    def step(self, batch, kind: str) -> dict:
        """Step function executed by lightning trainer module."""
        # run the model and calculate loss
        x, y_true = batch
        y_hat = self(x)

        if self.class_weights:
            weights = y_true * 0.75 + 0.75
        else:
            weights = None

        main_logit = y_hat[-1]
        main_loss = F.binary_cross_entropy_with_logits(main_logit, y_true, weight=weights)

        aux_losses = []
        for w, logit in zip(self.loss_cfg["per_layer"], y_hat[:-1]):
            aux_losses.append(w * F.binary_cross_entropy_with_logits(logit, y_true, weight=weights))
        aux_loss = sum(aux_losses) / max(1, len(aux_losses)) if aux_losses else torch.tensor(0.0, device=main_logit.device)

        loss = self.loss_cfg["main"] * main_loss + self.loss_cfg["aux"] * aux_loss

        if self.diversity_lambda > 0 and len(y_hat) > 1:
            div_loss = 0.0
            for i in range(1, len(y_hat)):
                diff = torch.sigmoid(y_hat[i]) - torch.sigmoid(y_hat[i - 1])
                div_loss += torch.exp(-torch.mean(diff ** 2))
            div_loss = div_loss / (len(y_hat) - 1)
            loss = loss + self.diversity_lambda * div_loss

        probs = torch.sigmoid(main_logit).view(-1)
        correct = ((probs > 0.5).flatten() == y_true.flatten()).sum()
        total = len(y_true)

        self.log(f"{kind}_main_loss", main_loss, batch_size=total)
        self.log(f"{kind}_aux_loss", aux_loss, batch_size=total)

        batch_dict = {
            "loss": loss,
            "correct": correct,
            "total": total,
            "probs": probs.detach(),
            "true": y_true.detach(),
        }
        return batch_dict

    def configure_optimizers(self):
        cfg = {"opt": "adam", "lr": self.lr, "wd": 0.0, "scheduler": "none"}
        cfg.update(self.optim_cfg)
        lr = cfg.get("lr", self.lr)
        wd = cfg.get("wd", 0.0)
        if cfg.get("opt", "adam") == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        scheduler_name = cfg.get("scheduler", "none")
        scheduler = None
        if scheduler_name == "cosine":
            t_max = cfg.get("t_max", 50)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        elif scheduler_name == "plateau":
            monitor = cfg.get("monitor", "val_loss")
            patience = cfg.get("patience", 10)
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience),
                "monitor": monitor,
            }
        elif scheduler_name == "onecycle":
            steps_per_epoch = cfg.get("steps_per_epoch")
            epochs = cfg.get("epochs", 1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs
            )

        warmup_steps = cfg.get("warmup_steps", 0)
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            if scheduler is not None:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, [warmup, scheduler], milestones=[warmup_steps]
                )
            else:
                scheduler = warmup

        if scheduler_name == "plateau" and isinstance(scheduler, dict):
            return [optimizer], [scheduler]
        elif scheduler is not None:
            return [optimizer], [scheduler]
        return [optimizer], []
