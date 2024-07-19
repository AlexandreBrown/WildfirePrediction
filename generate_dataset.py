import hydra
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from grid.square_meters_grid import SquareMetersGrid
from datasets.dataset_generator import DatasetGenerator
from pathlib import Path


logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="config", config_name="generate_dataset")
def main(cfg : DictConfig):
    
    boundary = CanadaBoundary(CanadaBoundaryDataSource(Path(cfg.boundaries.output_path)), target_epsg=cfg.projections.target_srs)
    boundary.load(provinces=cfg.boundaries.provinces)
    
    grid = SquareMetersGrid(
        pixel_size_in_meters=cfg.resolution.pixel_size_in_meters, 
        tile_size_in_pixels=cfg.resolution.tile_size_in_pixels
    )
    
    
    dynamic_sources = [ (source_name, source_values) for (source_name, source_values) in cfg.sources.get("dynamic", {}).items() ]
    static_sources = [ (source_name, source_values) for (source_name, source_values) in cfg.sources.get("static", {}).items() ]
    
    dataset_generator = DatasetGenerator(boundary, grid, input_folder_path=Path(cfg.paths.input_folder_path), output_folder_path=Path(cfg.paths.output_folder_path))
    
    dataset_generator.generate(
        dynamic_sources=dynamic_sources,
        static_sources=static_sources,
        periods_config=OmegaConf.to_container(cfg.periods),
        resolution_config=OmegaConf.to_container(cfg.resolution),
        projections_config=OmegaConf.to_container(cfg.projections)
    )


if __name__ == "__main__":
    main()
