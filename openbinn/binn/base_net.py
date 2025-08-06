"""Scaffolding for building PyTorch Lightning modules."""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

from typing import Tuple, List


class BaseNet(pl.LightningModule):
    """A basic scaffold for our modules, with default optimizer, scheduler, and loss
    function, and simple logging.
    """

    def __init__(self, lr: float = 0.01, scheduler: str = "lambda"):
        super().__init__()
        self.lr = lr
        self.scheduler = scheduler

        # Temporary storage for outputs during an epoch
        self.train_outputs = []
        self.val_outputs = []
        self.test_outputs = []

    def configure_optimizers(self) -> Tuple[List, List]:
        """Set up optimizers and schedulers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler == "lambda":
            lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.5 if epoch < 60 else 0.1
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif self.scheduler == "pnet":  # Take scheduler from pnet
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.25)
        else:
            scheduler = None

        return [optimizer], [scheduler] if scheduler else []

    def step(self, batch, kind: str) -> dict:
        """Generic step function that runs the network on a batch and outputs loss
        and accuracy information that will be aggregated at epoch end.

        This function is used to implement the training, validation, and test steps.
        """
        # run the model and calculate loss
        y_hat = self(batch)

        loss = F.nll_loss(y_hat, batch.y)

        # assess accuracy
        pred = y_hat.max(1)[1]
        correct = pred.eq(batch.y).sum().item()

        total = len(batch.y)

        batch_dict = {
            "loss": loss,
            "correct": correct,
            "total": total,
        }
        return batch_dict

    def epoch_end(self, outputs, kind: str):
        """Generic function for summarizing and logging the loss and accuracy over an
        epoch.

        Creates log entries with name `f"{kind}_loss"` and `f"{kind}_accuracy"`.
        """
        with torch.no_grad():
            total_loss = sum(_["loss"] * _["total"] for _ in outputs)
            total = sum(_["total"] for _ in outputs)
            avg_loss = total_loss / total

            correct = sum(_["correct"] for _ in outputs)
            avg_acc = correct / total

            if "probs" in outputs[0]:
                probs = torch.cat([_['probs'] for _ in outputs]).cpu().numpy()
                true = torch.cat([_['true'] for _ in outputs]).cpu().numpy()
                auc_val = roc_auc_score(true, probs)
            else:
                auc_val = None

        self.log(f"{kind}_loss", avg_loss)
        self.log(f"{kind}_accuracy", avg_acc)
        if auc_val is not None:
            self.log(f"{kind}_auc", auc_val)

    def training_step(self, batch, batch_idx) -> dict:
        output = self.step(batch, "train")
        self.train_outputs.append(output)  # Store outputs for epoch end
        return output

    def validation_step(self, batch, batch_idx) -> dict:
        output = self.step(batch, "val")
        self.val_outputs.append(output)  # Store outputs for epoch end
        return output

    def test_step(self, batch, batch_idx) -> dict:
        output = self.step(batch, "test")
        self.test_outputs.append(output)  # Store outputs for epoch end
        return output

    def on_train_epoch_end(self):
        # Process and log train epoch metrics
        self.epoch_end(self.train_outputs, "train")
        self.train_outputs = []  # Clear outputs for the next epoch

    def on_validation_epoch_end(self):
        # Process and log validation epoch metrics
        self.epoch_end(self.val_outputs, "val")
        self.val_outputs = []  # Clear outputs for the next epoch

    def on_test_epoch_end(self):
        # Process and log test epoch metrics
        self.epoch_end(self.test_outputs, "test")
        self.test_outputs = []  # Clear outputs for the next epoch
