from loguru import logger
from loggers.logger import Logger


class LoguruLogger(Logger):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch: int):
        displayable_metrics = self.get_displayable_metrics(self._epoch_metrics)
        logger.info(
            f"{self.stage_prefix}{self.epoch_log_prefix}{epoch}: {displayable_metrics}"
        )
        self._epoch_metrics.clear()

    def get_displayable_metrics(self, metrics: dict) -> str:
        return " | ".join([f"{k}={v:.4}" for k, v in metrics.items()])

    def on_step_end(self, step: int):
        displayable_metrics = self.get_displayable_metrics(self._step_metrics)
        logger.info(
            f"{self.stage_prefix}{self.step_log_prefix}{step}: {displayable_metrics}"
        )
        self._step_metrics.clear()
