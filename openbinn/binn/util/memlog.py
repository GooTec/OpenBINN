"""In-memory logger -- when you want to access your learning trajectory directly after
learning.
"""

import os
import torch

from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only

from argparse import Namespace
from typing import Optional, Union, List, Dict, Any, Mapping

import pandas as pd
from pathlib import Path


class ExperimentWriter:
    """In-memory experiment writer.

    Currently this supports logging hyperparameters and metrics.
    """

    def __init__(self):
        self.hparams: Dict[str, Any] = {}
        self.metrics: List[Dict[str, Union[float, int, str, Any]]] = []

    def log_hparams(self, params: Dict[str, Any]):
        """Record hyperparameters.

        This adds to previously recorded hparams, overwriting existing values in case
        of repeated keys.
        """
        self.hparams.update(params)

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Record metrics."""

        def _handle_value(value: Union[torch.Tensor, Any]) -> Any:
            if isinstance(value, torch.Tensor):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)  # Default to the next step
        elif not isinstance(step, int):
            raise ValueError("Step must be an integer.")

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)


class InMemoryLogger(Logger):
    """Log to in-memory storage.

    :param name: experiment name
    :param version: experiment version
    :param prefix: string to put at the beginning of metric keys
    :param save_dir: save directory (for checkpoints)
    :param df_aggregate_by_step: if true, the metrics dataframe is aggregated by step,
        with repeated non-NA entries averaged over
    :param hparams: access recorded hyperparameters
    :param metrics: access recorded metrics
    :param metrics_df: access metrics in Pandas dataframe format; this is only available
        after `finalize` or `save`
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        name: str = "lightning_logs",
        version: Union[int, str, None] = None,
        prefix: str = "",
        save_dir: str = "",
        df_aggregate_by_step: bool = True,
    ):
        super().__init__()
        self._name = name
        self._version = version
        self._prefix = prefix
        self._save_dir = save_dir
        self._metrics_df = None

        self.df_aggregate_by_step = df_aggregate_by_step

        self._experiment: Optional[ExperimentWriter] = None

    @property
    @rank_zero_experiment
    def experiment(self) -> ExperimentWriter:
        """Access the actual logger object."""
        if self._experiment is None:
            self._experiment = ExperimentWriter()

        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]):
        """Log hyperparameters."""
        params = _convert_params(params)
        self.experiment.log_hparams(params)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics, associating with given `step`."""
        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        self.experiment.log_metrics(metrics, step)

    def pandas(self) -> pd.DataFrame:
        """Return recorded metrics in a Pandas dataframe.

        By default the dataframe is aggregated by step, with repeated non-NA entries
        averaged over. This can be disabled using `self.df_aggregate_by_step`.
        """
        df = pd.DataFrame(self.experiment.metrics)
        if len(df) > 0:
            df.set_index("step", inplace=True)
            if self.df_aggregate_by_step:
                df = df.groupby("step").mean()

        return df

    def save(self):
        """Generate the metrics dataframe."""
        self._metrics_df = self.pandas()

    @property
    def name(self) -> str:
        """Experiment name."""
        return self._name

    @property
    def version(self) -> Union[int, str]:
        """Experiment version. Only used for checkpoints."""
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def root_dir(self) -> str:
        """Parent directory for all checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory
        is used and the checkpoint will be saved in `save_dir/version`.
        """
        return str(Path(self.save_dir) / self.name)

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named `'version_${self.version}'` but it can be overridden by
        passing a string value for the constructor's version parameter instead of `None`
        or an int.
        """
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        log_dir = Path(self.root_dir) / version
        return str(log_dir)

    @property
    def save_dir(self) -> str:
        """The current directory where checkpoints are saved."""
        return self._save_dir

    @property
    def hparams(self) -> Dict[str, Any]:
        """Access recorded hyperparameters.

        This is equivalent to `self.experiment.hparams`.
        """
        assert self._experiment is not None
        return self._experiment.hparams

    @property
    def metrics(self) -> List[Dict[str, float]]:
        """Access recorded metrics.

        This is equivalent to `self.experiment.metrics`.
        """
        assert self._experiment is not None
        return self._experiment.metrics

    @property
    def metrics_df(self) -> Optional[pd.DataFrame]:
        """Access recorded metrics in Pandas format.

        This is a cached version of the output from `self.pandas()` generated on
        `self.save()` or `self.finalize()`.
        """
        if self._metrics_df is None:
            raise ValueError("Metrics DataFrame has not been generated yet. Call `save()` first.")
        return self._metrics_df

    def _get_next_version(self) -> int:
        root_dir = Path(self.root_dir)

        if not root_dir.is_dir():
            return 0

        existing_versions = [
            int(d.name.split("_")[1])
            for d in root_dir.iterdir()
            if d.is_dir() and d.name.startswith("version_")
        ]

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


def _add_prefix(
    metrics: Mapping[str, Union[torch.Tensor, float]], prefix: str, separator: str
) -> Mapping[str, Union[torch.Tensor, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Code copied from PyTorch Lightning.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name
    Returns:
        Dictionary with prefix and separator inserted before each key
    """
    if prefix:
        metrics = {f"{prefix}{separator}{k}": v for k, v in metrics.items()}

    return metrics


def _convert_params(
    params: Optional[Union[Dict[str, Any], Namespace]]
) -> Dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Code copied from PyTorch Lightning.

    Args:
        params: Target to be converted to a dictionary
    Returns:
        params as a dictionary
    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params
