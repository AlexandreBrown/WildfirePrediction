import psutil
import sys
from pathlib import Path
from logging_utils.formats import default_project_format


def setup_logger(logger, run_name: str, debug: bool, enqueue: bool = False):
    logger.remove(0)

    log_folder_path = Path(f"logs/{run_name}/")
    log_folder_path.mkdir(parents=True, exist_ok=True)
    log_file_name = log_folder_path / "output.log"
    logger.add(
        str(log_file_name),
        format=default_project_format,
        colorize=True,
        level="DEBUG" if debug else "INFO",
        enqueue=enqueue,
    )
    logger.add(
        sys.stdout,
        format=default_project_format,
        colorize=True,
        level="DEBUG" if debug else "INFO",
        enqueue=enqueue,
    )


def get_ram_used():
    return lambda: psutil.virtual_memory().used / (1024.0**3)


def get_ram_total():
    return lambda: psutil.virtual_memory().total / (1024.0**3)
