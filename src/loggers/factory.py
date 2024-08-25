import os
from comet_ml import Experiment
from loggers.aggregate_logger import AggregateLogger
from loggers.logger import Logger
from loggers.loguru_logger import LoguruLogger
from loggers.cometml_logger import CometmlLogger


class LoggerFactory:
    def __init__(self, config: dict):
        for logger_name in config["logging"]["loggers"]:
            if logger_name.lower() == "cometml":
                COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
                COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
                COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
                self.cometml_experiment = Experiment(
                    api_key=COMET_ML_API_KEY,
                    project_name=COMET_ML_PROJECT_NAME,
                    workspace=COMET_ML_WORKSPACE,
                )
        self.config = config

    def create(self, stage_prefix: str) -> Logger:
        loggers = []
        for logger_name in self.config["logging"]["loggers"]:
            if logger_name.lower() == "loguru":
                new_logger = LoguruLogger()
            elif logger_name.lower() == "cometml":
                new_logger = CometmlLogger(self.cometml_experiment)
            else:
                raise ValueError(f"Unknown logger: {logger_name}")
            new_logger.stage_prefix = stage_prefix
            loggers.append(new_logger)

        return AggregateLogger(loggers)
