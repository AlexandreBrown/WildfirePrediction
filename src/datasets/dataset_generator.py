import uuid
import logging
import shutil
import geopandas as gpd
import multiprocessing as mp
import os
from boundaries.canada_boundary import CanadaBoundary
from grid.square_meters_grid import SquareMetersGrid
from preprocessing.data_aggregator import DataAggregator
from preprocessing.tiles import Tiles
from pathlib import Path
from collections import defaultdict


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
        except BaseException as e:
            logging.error(f"Error: {e}")
            shutil.rmtree(dataset_folder_path)
            raise e
        
        # self.generate_targets(target_years_range_reversed)
            
    def generate_data(
        self, 
        tmp_path: Path, 
        big_tiles: gpd.GeoDataFrame, 
        dynamic_sources: list, 
        static_sources: list, 
        periods_config: dict, 
        resolution_config: dict, 
        projections_config: dict
    ):
        max_nb_processes = max(1, len(os.sched_getaffinity(0)) - 1)
        
        logging.info(f"Max nb processes: {max_nb_processes}")
        
        processed_data_folder_path = tmp_path / Path("processed_data")
        
        sources_yearly_data_index = self.process_dynamic_data(processed_data_folder_path, dynamic_sources, big_tiles, periods_config, resolution_config, projections_config, max_nb_processes)

        self.process_static_data(processed_data_folder_path, sources_yearly_data_index, big_tiles, static_sources, resolution_config, projections_config)

        # TODO : Delete tiles that are not present for all sources and years

        # self.stack_data(dynamic_processed_data_paths, static_processed_data_paths)

    def process_static_data(
        self, 
        processed_data_folder_path: Path,
        sources_yearly_data_index: dict, 
        big_tiles: gpd.GeoDataFrame, 
        static_sources: list, 
        resolution_config: dict, 
        projections_config: dict
    ) -> list:
        logging.info(f"Processing {len(static_sources)} static sources...")
        
        for (source_name, source_values) in static_sources:
            layer_name = source_values['layer']
            
            raw_tiles_folder = Path(self.input_folder_path) / Path("static_data") / Path(f"{source_name}")
            
            tiling_output_folder_path = processed_data_folder_path / Path("static_data") / Path(f"{source_name}") / Path("tiles")
            
            logging.info(f"Processing {source_name}...")
            
            tiles = Tiles(
                raw_tiles_folder=raw_tiles_folder,
                layer_name=layer_name,
                tile_size_in_pixels=resolution_config['tile_size_in_pixels'],
                pixel_size_in_meters=resolution_config['pixel_size_in_meters'],
                output_folder=tiling_output_folder_path,
                big_tiles=big_tiles,
                source_srs=projections_config['source_srs'],
                target_srs=projections_config['target_srs'],
                resample_algorithm_continuous=resolution_config['resample_algorithm_continuous'],
                resample_algorithm_categorical=resolution_config['resample_algorithm_categorical']
            )
            
            tiles_paths = tiles.generate_preprocessed_tiles(data_type=source_values['data_type'])
            
            for year in sources_yearly_data_index.keys():
                sources_yearly_data_index[year][source_name] = tiles_paths

    def process_dynamic_data(
        self, 
        processed_data_folder_path: Path, 
        dynamic_sources: list, 
        big_tiles: gpd.GeoDataFrame, 
        periods_config: dict, 
        resolution_config: dict, 
        projections_config: dict,
        max_nb_processes: int
    ) -> dict:
        logging.info(f"Processing {len(dynamic_sources)} dynamic sources...")
                        
        source_args = [
            (processed_data_folder_path, source_name, source_values, source_values['layer'], source_values['is_affected_by_fires'], big_tiles, periods_config, resolution_config, projections_config) 
            for (source_name, source_values) in dynamic_sources
        ]
        
        nb_processes = min(max_nb_processes, len(source_args))
        
        with mp.Pool(processes=nb_processes) as pool:
            sources_yearly_data = pool.starmap(self.get_dynamic_data_yearly_data, source_args)

        formatted_sources_yearly_data_index = {}
                
        for source_yearly_data in sources_yearly_data:
            for year, source_yearly_data in source_yearly_data.items():
                source_name = list(source_yearly_data.keys())[0]
                source_yearly_data_paths = list(source_yearly_data.values())[0]
                if formatted_sources_yearly_data_index.get(year) is None:
                    formatted_sources_yearly_data_index[year] = {}
                
                formatted_sources_yearly_data_index[year][source_name] = source_yearly_data_paths
        
        return formatted_sources_yearly_data_index

    def get_dynamic_data_yearly_data(
        self, 
        processed_data_folder_path: Path, 
        source_name: str,
        source_values: dict,
        layer_name: str,
        is_affected_by_fires: bool,
        big_tiles: gpd.GeoDataFrame,
        periods_config: dict,
        resolution_config: dict, 
        projections_config: dict
    ) -> dict:
        year_start_inclusive, year_end_inclusive = self.get_dynamic_data_year_periods(periods_config, is_affected_by_fires)
        
        logging.info(f"{source_name} Year start: {year_start_inclusive}, year end: {year_end_inclusive}")
        
        years_tiles_data = []
        for year in range(year_start_inclusive, year_end_inclusive + 1):
            year_tiles_data = self.get_dynamic_source_tiles_months_data_for_1_year(processed_data_folder_path / Path(f"{year}"), year, source_name, layer_name, source_values, big_tiles, periods_config, resolution_config, projections_config)
            years_tiles_data.append(year_tiles_data)
            
        year_data_aggregator = DataAggregator()

        logging.info(f"{source_name} Aggregating data yearly...")
        
        yearly_data_for_source = {}

        for (year, tiles_months_data) in years_tiles_data:
            year_aggregated_output_folder_path = processed_data_folder_path / Path(f"{year}") / Path(f"{source_name}") / Path("year_aggregated_data")
            
            year_data_paths = []
            
            for tile_name, tile_months_data in tiles_months_data.items():
                if source_values['aggregate_by'] == 'average':
                    tile_year_data_path = year_data_aggregator.aggregate_files_by_average(
                        input_datasets_paths=tile_months_data,
                        output_folder_path=year_aggregated_output_folder_path
                    )
                elif source_values['aggregate_by'] == 'max':
                    tile_year_data_path = year_data_aggregator.aggregate_files_by_max(
                        input_datasets_paths=tile_months_data,
                        output_folder_path=year_aggregated_output_folder_path
                    )
                else:
                    raise ValueError(f"Unknown aggregation method: {source_values['aggregate_by']}")
                
                year_data_paths.append(tile_year_data_path)
            
            yearly_data_for_source[year] = {
                source_name: year_data_paths
            }
                
        return yearly_data_for_source
    
    def get_dynamic_data_year_periods(self, periods_config: dict, is_affected_by_fires: bool) -> tuple:
        if is_affected_by_fires:
            year_start_inclusive = periods_config['target_year_start_inclusive'] - periods_config['input_data_affected_by_fires_period_length_in_years']
            year_end_inclusive = periods_config['target_year_end_inclusive'] - periods_config['target_period_length_in_years']
        else:
            year_start_inclusive = periods_config['target_year_start_inclusive']
            year_end_inclusive = periods_config['target_year_end_inclusive']
        
        return year_start_inclusive, year_end_inclusive
    
    def get_dynamic_source_tiles_months_data_for_1_year(
        self,
        processed_data_year_output_folder_path: Path,
        year: int,
        source_name: str,
        layer_name: str,
        source_values: dict,
        big_tiles: gpd.GeoDataFrame,
        periods_config: dict,
        resolution_config: dict,
        projections_config: dict
    ) -> tuple:

        months = range(periods_config['month_start_inclusive'], periods_config['month_end_inclusive'] + 1)
        
        nb_months = len(list(months))
        
        logging.info(f"{source_name} Year: {year} Number of months to process: {nb_months}")
        
        logging.info(f"{source_name} Year: {year} Getting tiles for each month...")
        
        months_tiles = []
        for month in months:
            month_tiles = self.get_dynamic_source_tiles_month_data(processed_data_year_output_folder_path, year, month, source_name, layer_name, source_values, big_tiles, periods_config, resolution_config, projections_config)
            months_tiles.append(month_tiles)

        logging.info(f"{source_name} Year: {year} Aggregating data monthly...")
        
        month_data_aggregator = DataAggregator()
    
        tiles_months_data = defaultdict(list)
        
        for (tiles_path, tiles_output_path) in months_tiles:
            month_aggregated_output_folder_path = tiles_output_path / Path("month_aggregated_data")
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
                
                tiles_months_data[tile_name].append(tile_path_monthly_aggregated_data)
        
        return year, tiles_months_data
    
    def get_dynamic_source_tiles_month_data(self, processed_data_year_output_folder_path: Path, year: int, month: int, source_name: str, layer_name: str, source_values: dict, big_tiles: gpd.GeoDataFrame, periods_config: dict, resolution_config: dict, projections_config: dict) -> tuple:
        logging.info(f"{source_name} Year: {year} Month: {month}")
            
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
        
        return tiles_path, tiles_output_path
    
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
