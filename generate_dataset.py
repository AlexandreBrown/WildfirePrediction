import asyncio
import hydra
from loguru import logger
from omegaconf import DictConfig
from omegaconf import OmegaConf
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from grid.square_meters_grid import SquareMetersGrid
from datasets.dataset_generator import DatasetGenerator
from pathlib import Path
from logging_utils.logging import setup_logger


@hydra.main(version_base=None, config_path="config", config_name="generate_dataset")
def main(cfg: DictConfig):
    run_name = cfg["run"]["name"]
    debug = cfg["debug"]
    setup_logger(logger, run_name, debug, enqueue=True)
    logger.info(f"Run name: {run_name}")
    logger.info(f"Debug : {debug}")

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
