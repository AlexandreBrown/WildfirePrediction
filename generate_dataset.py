import asyncio
import hydra
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from grid.square_meters_grid import SquareMetersGrid
from datasets.dataset_generator import DatasetGenerator
from pathlib import Path
from preprocessing.no_data_value_preprocessor import NoDataValuePreprocessor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if len(logger.handlers) == 0:
    handler = logging.StreamHandler()
    logger.addHandler(handler)


@hydra.main(version_base=None, config_path="config", config_name="generate_dataset")
def main(cfg: DictConfig):

    logger.info(f"Debug : {cfg.debug}")

    canada_boundary = CanadaBoundary(
        CanadaBoundaryDataSource(Path(cfg.boundaries.output_path)),
        target_epsg=cfg.projections.target_srid,
    )
    canada_boundary.load(provinces=cfg.boundaries.provinces)

    grid = SquareMetersGrid(
        pixel_size_in_meters=cfg.resolution.pixel_size_in_meters,
        tile_size_in_pixels=cfg.resolution.tile_size_in_pixels,
    )

    dynamic_input_data = [
        (input_data_name, input_data_values)
        for (input_data_name, input_data_values) in cfg.input_data.get(
            "dynamic", {}
        ).items()
    ]
    static_input_data = [
        (input_data_name, input_data_values)
        for (input_data_name, input_data_values) in cfg.input_data.get(
            "static", {}
        ).items()
    ]

    no_data_fill_value = cfg.no_data_fill_value

    logger.info(f"No data fil value : {no_data_fill_value}")

    no_data_value_preprocessor = NoDataValuePreprocessor(no_data_fill_value)

    dataset_generator = DatasetGenerator(
        canada_boundary,
        grid,
        input_folder_path=Path(cfg.paths.input_folder_path),
        output_folder_path=Path(cfg.paths.output_folder_path),
        debug=cfg.debug,
        no_data_value_preprocessor=no_data_value_preprocessor,
        input_format=cfg.input_format,
        output_format=cfg.output_format,
    )

    asyncio.run(
        dataset_generator.generate(
            dynamic_input_data=dynamic_input_data,
            static_input_data=static_input_data,
            periods_config=OmegaConf.to_container(cfg.periods),
            resolution_config=OmegaConf.to_container(cfg.resolution),
            projections_config=OmegaConf.to_container(cfg.projections),
        )
    )


if __name__ == "__main__":
    main()
