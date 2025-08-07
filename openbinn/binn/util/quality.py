"""Various utility functions for assessing fit quality."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import roc_curve, auc, roc_auc_score
import torch.nn.functional as F
from typing import Iterable, Tuple, Optional
import pytorch_lightning as pl


def get_roc(
    model: torch.nn.Module, loader: Iterable, seed: Optional[int] = 1, exp: bool = True, takeLast: bool = False
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Run a model on the a dataset and calculate ROC and AUC.

    The model output values are by default exponentiated before calculating the ROC.
    XXX Why?

    When the model outputs a list of values instead of a tensor, the last element in the
    sequence is used by this function.

    :param model: model to test
    :param loader: data loader
    :param seed: PyTorch random seed; set to `None` to avoid setting the seed
    :param exp: if `True`, exponential model outputs before calculating ROC
    :param takeLast: Some architectures produce multiple outputs from intermediate layers.
                    If True, take the final prediction, if False take average of all predictions.
    :return: a tuple `(fpr, tpr, auc_value, ys, outs)`, where `(fpr, tpr)` are vectors
        representing the ROC curve; `auc_value` is the AUC; `ys` and `outs` are the
        expected (ground-truth) outputs and the (exponentiated) model outputs,
        respectively
    """
    if seed is not None:
        # keep everything reproducible!
        torch.manual_seed(seed)

    # make sure the model is in evaluation mode
    model.eval()

    outs = []
    ys = []
    device = next(iter(model.parameters())).device
    with torch.no_grad():
        for tb in loader:
            if hasattr(tb,"subject_id"):
                tb = tb.to(device)
                y=tb.y
            else:
                x, y = tb
                tb = x.to(device)
            
            output = model(tb)
    
            # handle multiple outputs
            if not torch.is_tensor(output):
                assert hasattr(output, "__getitem__")
                ## Either take last prediction
                if takeLast:
                    output = output[-1].cpu().numpy()
                ## Or average over all
                else:
                    # `output` can be a list of tensors on GPU. Move each to CPU
                    # before calculating the mean to avoid device mismatch
                    output = (
                        torch.stack([o.detach().to("cpu") for o in output])
                        .mean(dim=0)
                        .numpy()
                    )
            else:
                output = output.cpu().numpy()
                
            if exp:
                output = np.exp(output)
            outs.append(output)
    
            ys.append(y.cpu().numpy())

    outs = np.concatenate(outs)
    ys = np.concatenate(ys)
    if len(outs.shape) == 1 or outs.shape[0] == 1 or outs.shape[1] == 1:
        outs = np.column_stack([1 - outs, outs])
    fpr, tpr, _ = roc_curve(ys, outs[:, 1])
    auc_value = auc(fpr, tpr)

    return fpr, tpr, auc_value, ys, outs


def eval_metrics(model: torch.nn.Module, loader: Iterable) -> tuple[float, float, float]:
    """Return average loss, accuracy and AUC for a loader.

    The loss mirrors ``PNet.step`` so that recorded values match the training
    objective, including intermediate layer weighting and optional diversity
    penalty. Predictions for accuracy and AUC are based on the final layer.
    """
    device = next(iter(model.parameters())).device
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_probs = []
    all_true = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # compute loss using the model's step function without logging
            out_dict = model.step((x, y), "val", log=False)
            loss = out_dict["loss"]
            correct += out_dict["correct"].item() if isinstance(out_dict["correct"], torch.Tensor) else out_dict["correct"]
            total += out_dict["total"]
            total_loss += loss.item() * out_dict["total"]

            all_probs.append(out_dict["probs"].detach().cpu())
            all_true.append(out_dict["true"].detach().cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_true = torch.cat(all_true).numpy()
    auc_val = roc_auc_score(all_true, all_probs)
    acc = correct / total
    loss = total_loss / total
    return loss, acc, auc_val


class ValMetricsPrinter(pl.Callback):
    """Print validation metrics after each training epoch."""

    def __init__(self, val_loader: Iterable):
        super().__init__()
        self.val_loader = val_loader

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: torch.nn.Module) -> None:
        loss, acc, auc_val = eval_metrics(pl_module, self.val_loader)
        print(
            f"      Epoch {trainer.current_epoch + 1:03d}: "
            f"val_loss={loss:.4f} val_acc={acc:.4f} val_auc={auc_val:.4f}"
        )


class EpochMetricsPrinter(pl.Callback):
    """Print train and validation metrics after every epoch."""

    def __init__(self, train_loader: Iterable, val_loader: Iterable):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: torch.nn.Module) -> None:
        tr_loss, tr_acc, tr_auc = eval_metrics(pl_module, self.train_loader)
        va_loss, va_acc, va_auc = eval_metrics(pl_module, self.val_loader)
        print(
            f"      Epoch {trainer.current_epoch + 1:03d}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_auc={tr_auc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_auc={va_auc:.4f}"
        )


class MetricsRecorder(pl.Callback):
    """Record and plot metrics at the end of each epoch."""

    def __init__(self, out_dir: Path, train_loader: Iterable,
                 val_loader: Iterable, test_loader: Optional[Iterable] = None,
                 period: int = 1):
        super().__init__()
        self.out_dir = Path(out_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.period = period
        self.records: list[dict[str, float]] = []

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: torch.nn.Module) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.period != 0:
            return
        tr_loss, tr_acc, tr_auc = eval_metrics(pl_module, self.train_loader)
        va_loss, va_acc, va_auc = eval_metrics(pl_module, self.val_loader)
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_accuracy": tr_acc,
            "val_accuracy": va_acc,
            "train_auc": tr_auc,
            "val_auc": va_auc,
        }
        if self.test_loader is not None:
            te_loss, te_acc, te_auc = eval_metrics(pl_module, self.test_loader)
            rec.update({
                "test_loss": te_loss,
                "test_accuracy": te_acc,
                "test_auc": te_auc,
            })
        self.records.append(rec)

        msg = (
            f"Epoch {epoch:03d}: "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"train_acc={tr_acc:.4f} val_acc={va_acc:.4f} "
            f"train_auc={tr_auc:.4f} val_auc={va_auc:.4f}"
        )
        if self.test_loader is not None:
            msg += f" test_auc={rec['test_auc']:.4f}"
        print(msg)

    def on_train_end(self, trainer: pl.Trainer, pl_module: torch.nn.Module) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        perf_dir = self.out_dir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(perf_dir / "metrics.csv", index=False)

        vis_dir = self.out_dir / "visualize"
        vis_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(df["epoch"], df["train_loss"], label="train")
        plt.plot(df["epoch"], df["val_loss"], label="val")
        if "test_loss" in df:
            plt.plot(df["epoch"], df["test_loss"], label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(vis_dir / "loss_curve.png")
        plt.close()

        plt.figure()
        plt.plot(df["epoch"], df["train_accuracy"], label="train")
        plt.plot(df["epoch"], df["val_accuracy"], label="val")
        if "test_accuracy" in df:
            plt.plot(df["epoch"], df["test_accuracy"], label="test")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(vis_dir / "accuracy_curve.png")
        plt.close()

        plt.figure()
        plt.plot(df["epoch"], df["train_auc"], label="train")
        plt.plot(df["epoch"], df["val_auc"], label="val")
        if "test_auc" in df:
            plt.plot(df["epoch"], df["test_auc"], label="test")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(vis_dir / "auc_curve.png")
        plt.close()


class GradNormPrinter(pl.Callback):
    """Print gradient norm periodically during training."""

    def __init__(self, period: int = 100):
        super().__init__()
        self.period = period

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: torch.nn.Module,
        optimizer,
        opt_idx=None,
        *args,
        **kwargs,
    ) -> None:
        if trainer.global_step % self.period != 0:
            return
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm **= 0.5
        print(f"      Step {trainer.global_step}: grad_norm={total_norm:.4f}")

