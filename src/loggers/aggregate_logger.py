from loggers.logger import Logger


class AggregateLogger(Logger):
    def __init__(self, loggers: list):
        super().__init__()
        self.loggers = loggers

    @property
    def step_metrics(self):
        return self.loggers[0].step_metrics

    @property
    def epoch_metrics(self):
        return self.loggers[0].epoch_metrics

    def log_step_metric(self, metric_name: str, metric_value: float):
        for logger in self.loggers:
            logger.log_step_metric(metric_name, metric_value)

    def log_epoch_metric(self, metric_name: str, metric_value: float):
        for logger in self.loggers:
            logger.log_epoch_metric(metric_name, metric_value)

    def on_epoch_end(self, epoch: int):
        for logger in self.loggers:
            logger.on_epoch_end(epoch)

    def on_step_end(self, step: int):
        for logger in self.loggers:
            logger.on_step_end(step)
