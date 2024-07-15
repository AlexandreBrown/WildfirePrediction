import hydra
import logging
import geopandas as gpd
from omegaconf import DictConfig
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from grid.square_meters_grid import SquareMetersGrid
from preprocessing.data_aggregator import DataAggregator
from preprocessing.tiles import Tiles
from pathlib import Path


logging.basicConfig(level=logging.INFO)


def process_dynamic_data(cfg: DictConfig, tile_grids: gpd.GeoDataFrame):
    logging.info("Processing dynamic sources...")
    dynamic_sources = cfg.sources.dynamic
    for source_name, source_values in dynamic_sources.items():
        layer_name = source_values['layer']
        for year in range(cfg.periods.year_start_inclusive, cfg.periods.year_end_inclusive + 1):
            logging.info(f"Year: {year}")
            months_aggregated_data_paths = {}
            for month in range(cfg.periods.month_start_inclusive, cfg.periods.month_end_inclusive + 1):
                logging.info(f"Month: {month}")
                
                raw_tiles_folder = Path(cfg.paths.input_folder_path) / Path(f"{year}") / Path(f"{month}") / Path(f"{source_name}")
                
                year_output_folder_path = Path(cfg.paths.output_folder_path) / Path(f"{year}") / Path(f"{month}") / Path(f"{source_name}")
                
                tiling_output_folder_path = year_output_folder_path /  Path("tiling")
                
                logging.info(f"Processing {source_name}...")
                
                tiles = Tiles(
                    raw_tiles_folder=raw_tiles_folder,
                    layer_name=layer_name,
                    tile_size_in_pixels=cfg.resolution.tile_size_in_pixels,
                    pixel_size_in_meters=cfg.resolution.pixel_size_in_meters,
                    output_folder=tiling_output_folder_path,
                    tile_grids=tile_grids,
                    source_srs=cfg.projections.source_srs,
                    target_srs=cfg.projections.target_srs,
                    resample_algorithm_continuous=cfg.resolution.resample_algorithm_continuous,
                    resample_algorithm_categorical=cfg.resolution.resample_algorithm_categorical
                )
                
                tiles_path = tiles.generate_preprocessed_tiles(data_type=source_values['data_type'])
                
                month_aggregated_data_output_folder_path = year_output_folder_path / Path("month_aggregated_data")
                
                month_data_aggregator = DataAggregator()
                logging.info("Aggregating monthly data for tiles...")
                
                for tile_path in tiles_path:
                    if source_values['aggregate_by'] == 'average':
                        aggregated_tile_path = month_data_aggregator.aggregate_bands_by_average(
                            input_dataset_path=tile_path,
                            output_folder_path=month_aggregated_data_output_folder_path
                        )
                    elif source_values['aggregate_by'] == 'max':
                        aggregated_tile_path = month_data_aggregator.aggregate_bands_by_max(
                            input_dataset_path=tile_path,
                            output_folder_path=month_aggregated_data_output_folder_path
                        )
                    else:
                        raise ValueError(f"Unknown aggregation method: {source_values['aggregate_by']}")

                    if tile_path not in months_aggregated_data_paths.keys():
                        months_aggregated_data_paths[tile_path.stem] = [aggregated_tile_path]
                    else:
                        months_aggregated_data_paths[tile_path.stem].append(aggregated_tile_path)
            
            year_data_aggregator = DataAggregator()

            logging.info(f"Aggregating yearly data for {len(months_aggregated_data_paths)} tiles...")
            
            year_output_folder_path = Path(cfg.paths.output_folder_path) / Path(f"{year}") / Path(f"{source_name}") / Path("year_aggregated_data")
            
            for _, aggregated_tile_paths in months_aggregated_data_paths.items():
                if source_values['aggregate_by'] == 'average':
                    year_data_aggregator.aggregate_files_by_average(
                        input_datasets_paths=aggregated_tile_paths,
                        output_folder_path=year_output_folder_path
                    )
                elif source_values['aggregate_by'] == 'max':
                    year_data_aggregator.aggregate_files_by_max(
                        input_datasets_paths=aggregated_tile_paths,
                        output_folder_path=year_output_folder_path
                    )
                else:
                    raise ValueError(f"Unknown aggregation method: {source_values['aggregate_by']}")

def process_static_data(cfg: DictConfig, tile_grids: gpd.GeoDataFrame):
    logging.info("Processing static sources...")
    static_sources = cfg.sources.static
    for source_name, source_values in static_sources.items():
        layer_name = source_values['layer']
        
        raw_tiles_folder = Path(cfg.paths.input_folder_path) / Path("static_data") / Path(f"{source_name}")
        
        tiling_output_folder_path = Path(cfg.paths.output_folder_path) / Path(f"{source_name}") / Path("tiling")
        
        logging.info(f"Processing {source_name}...")
        
        tiles = Tiles(
            raw_tiles_folder=raw_tiles_folder,
            layer_name=layer_name,
            tile_size_in_pixels=cfg.resolution.tile_size_in_pixels,
            pixel_size_in_meters=cfg.resolution.pixel_size_in_meters,
            output_folder=tiling_output_folder_path,
            tile_grids=tile_grids,
            source_srs=cfg.projections.source_srs,
            target_srs=cfg.projections.target_srs,
            resample_algorithm_continuous=cfg.resolution.resample_algorithm_continuous,
            resample_algorithm_categorical=cfg.resolution.resample_algorithm_categorical
        )
        
        tiles.generate_preprocessed_tiles()


@hydra.main(version_base=None, config_path="config", config_name="generate_dataset")
def main(cfg : DictConfig):
    logging.info("Generating dataset...")
    
    boundary = CanadaBoundary(CanadaBoundaryDataSource(Path(cfg.boundaries.output_path)), target_epsg=cfg.projections.target_srs)
    boundary.load(provinces=cfg.boundaries.provinces)
    
    grid = SquareMetersGrid(
        pixel_size_in_meters=cfg.resolution.pixel_size_in_meters, 
        tile_size_in_pixels=cfg.resolution.tile_size_in_pixels
    )
    
    logging.info("Generating tile grids from boundary...")
    
    tile_grids = grid.get_tiles(boundary.boundary)
    
    if "dynamic" in cfg.sources:
        process_dynamic_data(cfg, tile_grids)
    
    if "static" in cfg.sources:
        process_static_data(cfg, tile_grids)

if __name__ == "__main__":
    main()
