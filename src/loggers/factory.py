import os
import copy
from loggers.aggregate_logger import AggregateLogger
from loggers.logger import Logger
from loggers.loguru_logger import LoguruLogger
from loggers.cometml_logger import CometmlLogger
from comet_ml import Experiment


class LoggerFactory:
    def __init__(self, config: dict):
        loggers: list = []
        for logger_name in config["logging"]["loggers"]:
            if logger_name.lower() == "loguru":
                loggers.append(LoguruLogger())
            elif logger_name.lower() == "cometml":
                COMET_ML_API_KEY = os.getenv("COMET_ML_API_KEY")
                COMET_ML_PROJECT_NAME = os.getenv("COMET_ML_PROJECT_NAME")
                COMET_ML_WORKSPACE = os.getenv("COMET_ML_WORKSPACE")
                experiment = Experiment(
                    api_key=COMET_ML_API_KEY,
                    project_name=COMET_ML_PROJECT_NAME,
                    workspace=COMET_ML_WORKSPACE,
                )
                loggers.append(CometmlLogger(experiment))

        self.loggers = loggers

    def create(self, stage_prefix: str) -> Logger:
        loggers_copy = copy.deepcopy(self.loggers)
        for logger in loggers_copy:
            logger.stage_prefix = stage_prefix
        return AggregateLogger(loggers_copy)
