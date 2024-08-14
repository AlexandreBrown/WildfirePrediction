import asyncio
import hydra
import sys
from datetime import datetime
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from grid.square_meters_grid import SquareMetersGrid
from datasets.dataset_generator import DatasetGenerator
from pathlib import Path
from logs_formatting.formats import default_project_format


@hydra.main(version_base=None, config_path="config", config_name="generate_dataset")
def main(cfg: DictConfig):
    logger.remove()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder_path = Path(f"logs/generate_dataset/{timestamp}/")
    log_folder_path.mkdir(parents=True, exist_ok=True)
    log_file_name = log_folder_path / "output.log"

    logger.add(
        str(log_file_name) if cfg.log_to_file else sys.stdout,
        format=default_project_format,
        colorize=True,
        level="DEBUG" if cfg.debug else "INFO",
        enqueue=True,
    )

    logger.info(f"Debug : {cfg.debug}")

    canada_boundary = CanadaBoundary(
        CanadaBoundaryDataSource(Path(cfg.boundaries.output_path)),
        target_epsg=cfg.projections.target_srid,
    )

    grid = SquareMetersGrid(
        pixel_size_in_meters=cfg.resolution.pixel_size_in_meters,
        tile_size_in_pixels=cfg.resolution.tile_size_in_pixels,
    )

    dataset_generator = DatasetGenerator(
        canada_boundary=canada_boundary,
        grid=grid,
        config=OmegaConf.to_container(cfg),
    )

    asyncio.run(dataset_generator.generate_dataset())


if __name__ == "__main__":
    main()
