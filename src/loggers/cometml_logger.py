from comet_ml import Experiment
from loggers.logger import Logger


class CometmlLogger(Logger):
    def __init__(
        self,
        experiment: Experiment,
    ):
        super().__init__()
        self.experiment = experiment

    def on_epoch_end(self, epoch: int) -> dict:
        epoch_metrics = super().on_epoch_end(epoch)
        self.experiment.log_metrics(
            epoch_metrics,
            prefix=f"{self.stage_prefix}{self.epoch_log_prefix[:-1]}",
            epoch=epoch,
        )
        self.experiment.log_epoch_end(epoch)
        return epoch_metrics

    def on_step_end(self, step: int) -> dict:
        step_metrics = super().on_step_end(step)
        self.experiment.log_metrics(
            step_metrics,
            prefix=f"{self.stage_prefix}{self.step_log_prefix[:-1]}",
            step=step,
        )
        return step_metrics

    def log_parameters(self, parameters: dict):
        self.experiment.log_parameters(parameters)
