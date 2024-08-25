import numpy as np
import copy
from abc import ABC, abstractmethod


class Logger(ABC):
    def __init__(self):
        self.step_log_prefix = "step_"
        self.epoch_log_prefix = "epoch_"
        self.stage_prefix = ""
        self._step_metrics = {}
        self._epoch_metrics = {}

    @property
    def step_metrics(self):
        return self._step_metrics

    @property
    def epoch_metrics(self):
        return self._epoch_metrics

    def log_step_metric(self, metric_name: str, metric_value: float):
        if self._step_metrics.get(metric_name) is None:
            self._step_metrics[metric_name] = [metric_value]
        else:
            self._step_metrics[metric_name].append(metric_value)

    def log_epoch_metric(self, metric_name: str, metric_value: float):
        if self._epoch_metrics.get(metric_name) is None:
            self._epoch_metrics[metric_name] = [metric_value]
        else:
            self._epoch_metrics[metric_name].append(metric_value)

    def on_epoch_end(self, epoch: int):
        self._epoch_metrics = {
            k: np.mean(v).item() for k, v in self._epoch_metrics.items()
        }
        epoch_metrics = copy.deepcopy(self._epoch_metrics)
        self._epoch_metrics.clear()
        return epoch_metrics

    def on_step_end(self, step: int):
        self._step_metrics = {
            k: np.mean(v).item() for k, v in self._step_metrics.items()
        }
        step_metrics = copy.deepcopy(self._step_metrics)
        self._step_metrics.clear()
        return step_metrics

    @abstractmethod
    def log_parameters(self, parameters: dict):
        pass

    def format_metrics(self, metrics: dict) -> dict:
        return {k: f"{v:.4}" for k, v in metrics.items()}
