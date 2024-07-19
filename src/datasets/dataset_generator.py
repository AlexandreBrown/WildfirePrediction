import uuid
import logging
import shutil
import geopandas as gpd
from boundaries.canada_boundary import CanadaBoundary
from grid.square_meters_grid import SquareMetersGrid
from preprocessing.data_aggregator import DataAggregator
from preprocessing.tiles import Tiles
from pathlib import Path


logging.basicConfig(level=logging.INFO)


class DatasetGenerator:
    def __init__(
        self, 
        boundary: CanadaBoundary, 
        grid: SquareMetersGrid,
        input_folder_path: Path,
        output_folder_path: Path
    ):
        self.boundary = boundary
        self.grid = grid
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        
    def generate(
        self,
        dynamic_sources: list,
        static_sources: list,
        periods_config: dict,
        resolution_config: dict,
        projections_config: dict
    ):
        logging.info("Generating dataset...")
        
        logging.info("Generating big tiles...")
        big_tiles = self.grid.get_tiles(self.boundary.boundary)
        
        logging.info("Generating dataset UUID...")
        dataset_uuid = str(uuid.uuid4())
        dataset_folder_path = self.output_folder_path / Path(dataset_uuid)
        dataset_folder_path.mkdir(parents=True, exist_ok=True)

        try:
            tmp_path = dataset_folder_path / Path("tmp")
            tmp_path.mkdir(parents=True, exist_ok=True)
        
            self.generate_data(tmp_path, big_tiles, dynamic_sources, static_sources, periods_config, resolution_config, projections_config)
        except Exception as e:
            logging.error(f"Error: {e}")
            shutil.rmtree(dataset_folder_path)
            raise e
        
        # self.generate_targets(target_years_range_reversed)
            
    def generate_data(self, tmp_path: Path, big_tiles: gpd.GeoDataFrame, dynamic_sources: list, static_sources: list, periods_config: dict, resolution_config: dict, projections_config: dict):
        dynamic_processed_data_paths = self.process_dynamic_data(tmp_path, big_tiles, dynamic_sources, periods_config, resolution_config, projections_config)
        
        # static_processed_data_paths = self.process_static_data(big_tiles, static_sources, resolution_config, projections_config)

        # self.stack_data(dynamic_processed_data_paths, static_processed_data_paths)
        

    def process_dynamic_data(self, tmp_path: Path, big_tiles: gpd.GeoDataFrame, dynamic_sources: list, periods_config: dict, resolution_config: dict, projections_config: dict) -> list:
        logging.info("Processing dynamic sources...")
        
        processed_data_folder_path = tmp_path / Path("processed_data")
        
        for (source_name, source_values) in dynamic_sources:
            logging.info(f"Processing {source_name}...")
            layer_name = source_values['layer']
            is_affected_by_fires = source_values['is_affected_by_fires']
            
            if is_affected_by_fires:
                year_start_inclusive = periods_config['target_year_start_inclusive'] - periods_config['input_data_affected_by_fires_period_length_in_years']
                year_end_inclusive = periods_config['target_year_end_inclusive'] - 1
            else:
                year_start_inclusive = periods_config['target_year_start_inclusive']
                year_end_inclusive = periods_config['target_year_end_inclusive']
            
            for year in range(year_start_inclusive, year_end_inclusive, 1):
                logging.info(f"Year: {year}")
                tiles_months_aggregated_data_paths = {}
                processed_data_year_output_folder_path = processed_data_folder_path / Path(f"{year}")
                for month in range(periods_config['month_start_inclusive'], periods_config['month_end_inclusive'] + 1):
                    logging.info(f"Month: {month}")
                    
                    raw_tiles_folder = self.input_folder_path / Path(f"{year}") / Path(f"{month}") / Path(f"{source_name}")
                    
                    tiles_output_path = processed_data_year_output_folder_path / Path(f"{month}") / Path(f"{source_name}")
                    
                    tiles = Tiles(
                        raw_tiles_folder=raw_tiles_folder,
                        layer_name=layer_name,
                        tile_size_in_pixels=resolution_config['tile_size_in_pixels'],
                        pixel_size_in_meters=resolution_config['pixel_size_in_meters'],
                        output_folder=tiles_output_path,
                        big_tiles=big_tiles,
                        source_srs=projections_config['source_srs'],
                        target_srs=projections_config['target_srs'],
                        resample_algorithm_continuous=resolution_config['resample_algorithm_continuous'],
                        resample_algorithm_categorical=resolution_config['resample_algorithm_categorical']
                    )
                    
                    tiles_path = tiles.generate_preprocessed_tiles(data_type=source_values['data_type'])
                    
                    month_aggregated_output_folder_path = tiles_output_path / Path("month_aggregated_data")
                    
                    month_data_aggregator = DataAggregator()
                    logging.info("Aggregating data monthly...")
                    
                    for tile_path in tiles_path:
                        if source_values['aggregate_by'] == 'average':
                            tile_path_monthly_aggregated_data = month_data_aggregator.aggregate_bands_by_average(
                                input_dataset_path=tile_path,
                                output_folder_path=month_aggregated_output_folder_path
                            )
                        elif source_values['aggregate_by'] == 'max':
                            tile_path_monthly_aggregated_data = month_data_aggregator.aggregate_bands_by_max(
                                input_dataset_path=tile_path,
                                output_folder_path=month_aggregated_output_folder_path
                            )
                        else:
                            raise ValueError(f"Unknown aggregation method: {source_values['aggregate_by']}")

                        tile_name = tile_path.stem
                        
                        if tile_name not in tiles_months_aggregated_data_paths.keys():
                            tiles_months_aggregated_data_paths[tile_name] = [tile_path_monthly_aggregated_data]
                        else:
                            tiles_months_aggregated_data_paths[tile_name].append(tile_path_monthly_aggregated_data)
                
                year_data_aggregator = DataAggregator()

                logging.info("Aggregating data yearly...")
                
                year_aggregated_output_folder_path = processed_data_year_output_folder_path / Path(f"{source_name}") / Path("year_aggregated_data")
                
                for _, aggregated_tile_paths in tiles_months_aggregated_data_paths.items():
                    if source_values['aggregate_by'] == 'average':
                        year_data_aggregator.aggregate_files_by_average(
                            input_datasets_paths=aggregated_tile_paths,
                            output_folder_path=year_aggregated_output_folder_path
                        )
                    elif source_values['aggregate_by'] == 'max':
                        year_data_aggregator.aggregate_files_by_max(
                            input_datasets_paths=aggregated_tile_paths,
                            output_folder_path=year_aggregated_output_folder_path
                        )
                    else:
                        raise ValueError(f"Unknown aggregation method: {source_values['aggregate_by']}")
    
    def process_static_data(big_tiles: gpd.GeoDataFrame, static_sources: list, resolution_config: dict, projections_config: dict) -> list:
        return []
    """
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
            big_tiles=big_tiles,
            source_srs=cfg.projections.source_srs,
            target_srs=cfg.projections.target_srs,
            resample_algorithm_continuous=cfg.resolution.resample_algorithm_continuous,
            resample_algorithm_categorical=cfg.resolution.resample_algorithm_categorical
        )
        
        tiles.generate_preprocessed_tiles()

    """
    
    def generate_target_years_ranges(self, periods_config: dict) -> list:
        target_year_start_inclusive = periods_config['year_start_inclusive']
        target_year_end_inclusive = periods_config['year_end_inclusive']
        target_period_length_in_years = periods_config['target_period_length_in_years']
        target_year_ranges = []
        
        for target_year_end in range(target_year_end_inclusive, target_year_start_inclusive - 1, -1):
            target_year_start = target_year_end - target_period_length_in_years + 1
            if target_year_start < target_year_start_inclusive:
                continue
            target_year_ranges.append(range(target_year_start, target_year_end))
        
        return target_year_ranges
    
    def stack_data(dynamic_processed_data_paths: list, static_processed_data_paths: list):
        pass
    
    def generate_targets(self):
        pass
