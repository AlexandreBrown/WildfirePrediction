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
        self._step_metrics[metric_name] = metric_value

    def log_epoch_metric(self, metric_name: str, metric_value: float):
        self._epoch_metrics[metric_name] = metric_value

    @abstractmethod
    def on_epoch_end(self, epoch: int):
        pass

    @abstractmethod
    def on_step_end(self, step: int):
        pass

    @abstractmethod
    def log_parameters(self, parameters: dict):
        pass

    def format_metrics(self, metrics: dict) -> dict:
        return {k: f"{v:.4}" for k, v in metrics.items()}
