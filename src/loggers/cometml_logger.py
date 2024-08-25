from comet_ml import Experiment
from loggers.logger import Logger


class CometmlLogger(Logger):
    def __init__(
        self,
        experiment: Experiment,
    ):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch: int):
        self.experiment.log_metrics(
            self._epoch_metrics,
            prefix=f"{self.stage_prefix}{self.epoch_log_prefix}",
            epoch=epoch,
        )
        self._epoch_metrics.clear()
        self.experiment.log_epoch_end(epoch)

    def on_step_end(self, step: int):
        self.experiment.log_metrics(
            self._step_metrics,
            prefix=f"{self.stage_prefix}{self.step_log_prefix}",
            step=step,
        )
        self._step_metrics.clear()

    def log_parameters(self, parameters: dict):
        self.experiment.log_parameters(parameters)
