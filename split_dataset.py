import hydra
import json
import torch
import numpy as np
from datasets.wildfire_data_module import WildfireDataModule
from pathlib import Path
from loguru import logger
from omegaconf import DictConfig
from logging_utils.logging import setup_logger


@hydra.main(version_base=None, config_path="config", config_name="split_dataset")
def main(cfg: DictConfig):
    run_name = cfg["run"]["name"]
    debug = cfg["debug"]
    setup_logger(logger, run_name, debug)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Debug : {debug}")

    output_folder_path = Path(cfg["data"]["output_folder_path"]) / Path(
        cfg["run"]["name"]
    )
    output_folder_path.mkdir(parents=True, exist_ok=True)

    train_periods = get_train_periods(cfg)

    input_data_periods_folders_paths = [
        Path(p) for p in cfg["data"]["input_data_periods_folders_paths"]
    ]
    target_periods_folders_paths = [
        Path(p) for p in cfg["data"]["target_periods_folders_paths"]
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info(f"Seed: {cfg.seed}")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    data_module = WildfireDataModule(
        input_data_indexes_to_remove=[],
        seed=cfg["seed"],
        model_input_size=cfg["model"]["input_resolution_in_pixels"],
        input_data_periods_folders_paths=input_data_periods_folders_paths,
        target_periods_folders_paths=target_periods_folders_paths,
        train_periods=train_periods,
        val_split=cfg["training"]["val_split"],
        preprocessing_num_workers=cfg["data"]["preprocessing_num_workers"],
        output_folder_path=output_folder_path,
        device=device,
    )

    data_split_info = data_module.split_data()

    info_file_path = output_folder_path / "data_split_info.json"
    with open(info_file_path, "w") as f:
        json.dump(data_split_info, f, indent=4)

    logger.info(f"Data split info saved to {str(info_file_path)}")


def get_train_periods(cfg: DictConfig) -> list:
    target_year_start_inclusive = cfg["training"]["train_periods"]["start_inclusive"]
    target_year_end_inclusive = cfg["training"]["train_periods"]["end_inclusive"]
    target_period_length_in_years = cfg["training"]["train_periods"][
        "period_length_in_years"
    ]
    target_years_ranges = []

    for target_year_start in range(
        target_year_start_inclusive, target_year_end_inclusive + 1, 1
    ):
        target_year_end = target_year_start + target_period_length_in_years - 1
        assert (
            target_year_end <= target_year_end_inclusive
        ), f"Target year end {target_year_end} is greater than target year end inclusive {target_year_end_inclusive}"
        target_years_ranges.append(range(target_year_start, target_year_end + 1))

    logger.info(f"Train target years ranges: {target_years_ranges}")

    return target_years_ranges


if __name__ == "__main__":
    main()
