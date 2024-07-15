import hydra
import logging
from omegaconf import DictConfig
from boundaries.canada_boundary import CanadaBoundary
from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from preprocessing.data_aggregator import DataAggregator
from preprocessing.tiles import Tiles
from pathlib import Path


logging.basicConfig(level=logging.INFO)


def process_dynamic_data(cfg: DictConfig, boundary: CanadaBoundary):
    logging.info("Processing dynamic sources...")
    dynamic_sources = cfg.sources.dynamic
    for source_name, source_values in dynamic_sources.items():
        layer_name = source_values['layer']
        for year in range(cfg.periods.year_start_inclusive, cfg.periods.year_end_inclusive + 1):
            logging.info(f"Year: {year}")
            months_aggregated_data_paths = []
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
                    boundary=boundary,
                    source_srs=cfg.projections.source_srs,
                    target_srs=cfg.projections.target_srs,
                    resample_algorithm=cfg.resolution.resample_algorithm
                )
                
                tiles_path = tiles.generate_preprocessed_tiles()
                
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
                    
                    months_aggregated_data_paths.append(aggregated_tile_path)
            
            year_data_aggregator = DataAggregator()

            logging.info(f"Aggregating yearly data for {len(months_aggregated_data_paths)} tiles...")
            
            year_output_folder_path = Path(cfg.paths.output_folder_path) / Path(f"{year}") / Path(f"{source_name}") / Path("year_aggregated_data")
            
            if source_values['aggregate_by'] == 'average':
                year_data_aggregator.aggregate_files_by_average(
                    input_datasets_paths=months_aggregated_data_paths,
                    output_folder_path=year_output_folder_path
                )
            elif source_values['aggregate_by'] == 'max':
                year_data_aggregator.aggregate_files_by_max(
                    input_datasets_paths=months_aggregated_data_paths,
                    output_folder_path=year_output_folder_path
                )
            else:
                raise ValueError(f"Unknown aggregation method: {source_values['aggregate_by']}")
    

def process_static_data(cfg: DictConfig, boundary: CanadaBoundary):
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
            boundary=boundary,
            source_srs=cfg.projections.source_srs,
            target_srs=cfg.projections.target_srs,
            resample_algorithm=cfg.resolution.resample_algorithm
        )
        
        tiles.generate_preprocessed_tiles()


@hydra.main(version_base=None, config_path="config", config_name="generate_dataset")
def main(cfg : DictConfig):
    logging.info("Generating dataset...")
    
    boundary = CanadaBoundary(CanadaBoundaryDataSource(Path(cfg.boundaries.output_path)), target_epsg=cfg.projections.target_srs)
    boundary.load(provinces=cfg.boundaries.provinces)
    
    if "dynamic" in cfg.sources:
        process_dynamic_data(cfg, boundary)
    
    if "static" in cfg.sources:
        process_static_data(cfg, boundary)

if __name__ == "__main__":
    main()
