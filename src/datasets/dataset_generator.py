import uuid
import logging
import shutil
import geopandas as gpd
import os
import json
from boundaries.canada_boundary import CanadaBoundary
from data_sources.nbac_fire_data_source import NbacFireDataSource
from grid.square_meters_grid import SquareMetersGrid
from preprocessing.data_aggregator import DataAggregator
from preprocessing.no_data_value_preprocessor import NoDataValuePreprocessor
from preprocessing.tiles_preprocessor import TilesPreprocessor
from pathlib import Path
from collections import defaultdict
from osgeo import gdal
from targets.fire_occurrence_target import FireOccurrenceTarget


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class DatasetGenerator:
    def __init__(
        self, 
        canada_boundary: CanadaBoundary, 
        grid: SquareMetersGrid,
        input_folder_path: Path,
        output_folder_path: Path,
        debug: bool,
        no_data_value_preprocessor: NoDataValuePreprocessor
    ):
        self.canada_boundary = canada_boundary
        self.grid = grid
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.debug = debug
        self.no_data_value_preprocessor = no_data_value_preprocessor
    
    def generate(
        self,
        dynamic_input_data: list,
        static_input_data: list,
        periods_config: dict,
        resolution_config: dict,
        projections_config: dict
    ):
        logger.info("Generating dataset...")
        
        logger.info("Generating big tiles boundaries...")
        big_tiles_boundaries = self.grid.get_tiles_boundaries(self.canada_boundary.boundary)
        logger.info(f"Generated {len(big_tiles_boundaries)} big tiles boundaries!")
        
        dataset_folder_path = self.get_dataset_folder_path()

        try:
            target_years_ranges = self.generate_target_years_ranges(periods_config)
            
            tmp_path = self.get_dataset_tmp_folder_path(dataset_folder_path)
            
            self.process_input_data(dataset_folder_path, tmp_path, big_tiles_boundaries, dynamic_input_data, static_input_data, periods_config, resolution_config, projections_config, target_years_ranges)
            
            self.generate_targets(dataset_folder_path, tmp_path, target_years_ranges, big_tiles_boundaries, resolution_config, projections_config)
            
            self.cleanup_tmp_folder()
        except BaseException as e:
            logger.error(f"Error: {e}")
            self.cleanup_dataset_folder(dataset_folder_path)
            raise e
    
    def get_dataset_folder_path(self) -> Path:
        dataset_uuid = self.get_dataset_uuid()
        
        dataset_folder_path = self.output_folder_path / Path(dataset_uuid)
        dataset_folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset folder path : {str(dataset_folder_path)}")
        
        return dataset_folder_path
    
    def get_dataset_uuid(self) -> str:
        ds_uuid = str(uuid.uuid4())
        logger.info(f"Dataset UUID: {ds_uuid}")
        return ds_uuid
    
    def get_dataset_tmp_folder_path(self, dataset_folder_path: Path) -> Path:
        tmp_folder_path = dataset_folder_path / Path("tmp")
        tmp_folder_path.mkdir(parents=True, exist_ok=True)
        return tmp_folder_path

    def generate_target_years_ranges(self, periods_config: dict) -> list:
        target_year_start_inclusive = periods_config['target_year_start_inclusive']
        target_year_end_inclusive = periods_config['target_year_end_inclusive']
        target_period_length_in_years = periods_config['target_period_length_in_years']
        target_years_ranges = []
        
        for target_year_start in range(target_year_start_inclusive, target_year_end_inclusive + 1, 1):
            target_year_end = target_year_start + target_period_length_in_years - 1
            assert target_year_end <= target_year_end_inclusive, f"Target year end {target_year_end} is greater than target year end inclusive {target_year_end_inclusive}"
            target_years_ranges.append(range(target_year_start, target_year_end + 1))
        
        logger.info(f"Target years ranges: {target_years_ranges}")
        
        return target_years_ranges

    def process_input_data(
        self, 
        dataset_folder_path: Path, 
        tmp_path: Path,
        big_tiles_boundaries: gpd.GeoDataFrame, 
        dynamic_sources: list, 
        static_sources: list, 
        periods_config: dict, 
        resolution_config: dict, 
        projections_config: dict,
        target_years_ranges: list
    ):
        logger.info("Processing input data...")
        
        processed_data_folder_path = tmp_path / Path("processed_input_data")
        
        processed_data_folder_path.mkdir(parents=True, exist_ok=True)
        
        input_data_yearly_data_index = self.process_dynamic_input_data(processed_data_folder_path, dynamic_sources, big_tiles_boundaries, periods_config, resolution_config, projections_config)

        input_data_yearly_data_index = self.process_static_data(processed_data_folder_path, input_data_yearly_data_index, big_tiles_boundaries, static_sources, resolution_config, projections_config)

        tiles_names = self.get_tiles_names(input_data_yearly_data_index)

        self.stack_data(input_data_yearly_data_index, tiles_names, dataset_folder_path, periods_config, resolution_config, dynamic_sources, static_sources, target_years_ranges)
        
        self.log_input_data_processing_results(dataset_folder_path, input_data_yearly_data_index)
    
    def process_dynamic_input_data(
        self, 
        processed_data_folder_path: Path, 
        dynamic_input_data: list, 
        big_tiles_boundaries: gpd.GeoDataFrame, 
        periods_config: dict, 
        resolution_config: dict, 
        projections_config: dict
    ) -> dict:
        logger.info(f"Processing {len(dynamic_input_data)} dynamic input data...")
                        
        input_data_yearly_data = []
        for (input_data_name, input_data_values) in dynamic_input_data:
            logger.info(f"Processing {input_data_name}...")
            input_data_yearly_data = self.get_dynamic_input_data_yearly_data(processed_data_folder_path, input_data_name, input_data_values, input_data_values['layer'], input_data_values['is_affected_by_fires'], big_tiles_boundaries, periods_config, resolution_config, projections_config)
            input_data_yearly_data.append(input_data_yearly_data)

        logger.info("Formatting input data yearly data index...")
        formatted_input_data_yearly_data_index = {}
                
        for input_data_yearly_data in input_data_yearly_data:
            for year, input_data_yearly_data in input_data_yearly_data.items():
                current_input_data_name = list(input_data_yearly_data.keys())[0]
                current_source_yearly_data_paths = list(input_data_yearly_data.values())[0]
                
                if formatted_input_data_yearly_data_index.get(year) is None:
                    formatted_input_data_yearly_data_index[year] = {}
                
                formatted_input_data_yearly_data_index[year][current_input_data_name] = current_source_yearly_data_paths
        
        return formatted_input_data_yearly_data_index

    def get_dynamic_input_data_yearly_data(
        self, 
        processed_data_folder_path: Path, 
        input_data_name: str,
        input_data_values: dict,
        layer_name: str,
        is_affected_by_fires: bool,
        big_tiles_boundaries: gpd.GeoDataFrame,
        periods_config: dict,
        resolution_config: dict, 
        projections_config: dict
    ) -> dict:
        year_range = self.get_dynamic_input_data_total_years_extent(periods_config, is_affected_by_fires)
        
        logger.info(f"{input_data_name}: Year range {year_range}")
        
        months_data_for_each_year = []
        for year in year_range:
            months_data_for_1_year = self.get_dynamic_input_data_tiles_months_data_for_1_year(processed_data_folder_path / Path(f"{year}"), year, input_data_name, layer_name, input_data_values, big_tiles_boundaries, periods_config, resolution_config, projections_config)
            months_data_for_each_year.append(months_data_for_1_year)
            
        return self.aggregate_dynamic_input_data_yearly(input_data_name, input_data_values, months_data_for_each_year, processed_data_folder_path)

    def get_dynamic_input_data_total_years_extent(self, periods_config: dict, is_affected_by_fires: bool) -> range:
        if is_affected_by_fires:
            year_end_inclusive = periods_config['target_year_start_inclusive'] - 1
            year_start_inclusive = year_end_inclusive - periods_config['input_data_affected_by_fires_period_length_in_years'] + 1
        else:
            year_end_inclusive = periods_config['target_year_end_inclusive']
            year_start_inclusive = periods_config['target_year_start_inclusive']
        
        return range(year_start_inclusive, year_end_inclusive+1)

    def get_dynamic_input_data_tiles_months_data_for_1_year(
        self,
        processed_data_year_output_folder_path: Path,
        year: int,
        input_data_name: str,
        layer_name: str,
        input_data_values: dict,
        big_tiles_boundaries: gpd.GeoDataFrame,
        periods_config: dict,
        resolution_config: dict,
        projections_config: dict
    ) -> tuple:

        months_range = range(periods_config['month_start_inclusive'], periods_config['month_end_inclusive'] + 1)
        
        logger.info(f"{input_data_name}: Year: {year} Getting tiles for each month for range {months_range}...")
        
        months_tiles = []
        for month in months_range:
            month_tiles = self.get_dynamic_input_data_tiles_for_1_month(processed_data_year_output_folder_path, year, month, input_data_name, layer_name, input_data_values, big_tiles_boundaries, resolution_config, projections_config)
            months_tiles.append(month_tiles)

        tiles_months_data = self.aggregate_dynamic_input_data_monthly(months_tiles, input_data_values)
        
        return year, tiles_months_data
    
    def get_dynamic_input_data_tiles_for_1_month(self, processed_data_year_output_folder_path: Path, year: int, month: int, input_data_name: str, layer_name: str, input_data_values: dict, big_tiles_boundaries: gpd.GeoDataFrame, resolution_config: dict, projections_config: dict) -> tuple:
        logger.info(f"{input_data_name}: Year: {year} Month: {month}")
            
        raw_tiles_folder = self.input_folder_path / Path(f"{year}") / Path(f"{month}") / Path(f"{input_data_name}")
        
        tiles_output_path = processed_data_year_output_folder_path / Path(f"{month}") / Path(f"{input_data_name}")
        
        tiles = TilesPreprocessor(
            raw_tiles_folder=raw_tiles_folder,
            layer_name=layer_name,
            tile_size_in_pixels=resolution_config['tile_size_in_pixels'],
            pixel_size_in_meters=resolution_config['pixel_size_in_meters'],
            output_folder=tiles_output_path,
            big_tiles_boundaries=big_tiles_boundaries,
            source_srs=projections_config['source_srs'],
            target_srs=projections_config['target_srs'],
            resample_algorithm_continuous=resolution_config['resample_algorithm_continuous'],
            resample_algorithm_categorical=resolution_config['resample_algorithm_categorical']
        )
        
        logger.info(f"{input_data_name}: Preprocessing tiles for year {year} and month {month}...")
        tiles_path = tiles.preprocess_tiles(data_type=input_data_values['data_type'])
        logger.info(f"{input_data_name}: Generated {len(tiles_path)} tiles for year {year} and month {month}!")
        
        return tiles_path, tiles_output_path
    
    def aggregate_dynamic_input_data_monthly(self, input_data_name: str, year: int, months_tiles: list, input_data_values: dict) -> dict:
        logger.info(f"{input_data_name}: Year: {year} Aggregating data monthly...")
        
        month_data_aggregator = DataAggregator()
    
        tiles_months_data = defaultdict(list)
        
        for (tiles_path, tiles_output_path) in months_tiles:
            month_aggregated_output_folder_path = tiles_output_path / Path("month_aggregated_data")
            for tile_path in tiles_path:
                if input_data_values['aggregate_by'] == 'average':
                    tile_path_monthly_aggregated_data = month_data_aggregator.aggregate_bands_by_average(
                        input_dataset_path=tile_path,
                        output_folder_path=month_aggregated_output_folder_path
                    )
                elif input_data_values['aggregate_by'] == 'max':
                    tile_path_monthly_aggregated_data = month_data_aggregator.aggregate_bands_by_max(
                        input_dataset_path=tile_path,
                        output_folder_path=month_aggregated_output_folder_path
                    )
                else:
                    raise ValueError(f"Unknown aggregation method: {input_data_values['aggregate_by']}")

                tile_name = tile_path.stem
                
                tiles_months_data[tile_name].append(tile_path_monthly_aggregated_data)
        
        return tiles_months_data

    def aggregate_dynamic_input_data_yearly(self, input_data_name: str, input_data_values: dict, months_data_for_each_year: list, processed_data_folder_path: Path) -> dict:
        year_data_aggregator = DataAggregator()

        logger.info(f"{input_data_name} Aggregating data yearly...")
        
        yearly_data = {}

        for (year, tiles_months_data) in months_data_for_each_year:
            year_aggregated_output_folder_path = processed_data_folder_path / Path(f"{year}") / Path(f"{input_data_name}") / Path("year_aggregated_data")
            
            year_data_paths = []
            
            for tile_name, tile_months_data in tiles_months_data.items():
                if input_data_values['aggregate_by'] == 'average':
                    tile_year_data_path = year_data_aggregator.aggregate_files_by_average(
                        input_datasets_paths=tile_months_data,
                        output_folder_path=year_aggregated_output_folder_path
                    )
                elif input_data_values['aggregate_by'] == 'max':
                    tile_year_data_path = year_data_aggregator.aggregate_files_by_max(
                        input_datasets_paths=tile_months_data,
                        output_folder_path=year_aggregated_output_folder_path
                    )
                else:
                    raise ValueError(f"Unknown aggregation method: {input_data_values['aggregate_by']}")
                
                year_data_paths.append(tile_year_data_path)
            
            yearly_data[year] = {
                input_data_name: year_data_paths
            }
        
        return yearly_data

    def process_static_data(
        self, 
        processed_data_folder_path: Path,
        input_data_yearly_data_index: dict, 
        big_tiles_boundaries: gpd.GeoDataFrame, 
        static_input_data: list, 
        resolution_config: dict, 
        projections_config: dict
    ) -> dict:
        logger.info(f"Processing {len(static_input_data)} static input data...")
        
        for (input_data_name, input_data_values) in static_input_data:
            layer_name = input_data_values['layer']
            
            static_input_data_folder_name = "static_data"
            
            raw_tiles_folder = Path(self.input_folder_path) / Path(static_input_data_folder_name) / Path(f"{input_data_name}")
            
            tiling_output_folder_path = processed_data_folder_path / Path(static_input_data_folder_name) / Path(f"{input_data_name}") / Path("tiles")
            
            logger.info(f"Processing {input_data_name}...")
            
            tiles = TilesPreprocessor(
                raw_tiles_folder=raw_tiles_folder,
                layer_name=layer_name,
                tile_size_in_pixels=resolution_config['tile_size_in_pixels'],
                pixel_size_in_meters=resolution_config['pixel_size_in_meters'],
                output_folder=tiling_output_folder_path,
                big_tiles_boundaries=big_tiles_boundaries,
                source_srs=projections_config['source_srs'],
                target_srs=projections_config['target_srs'],
                resample_algorithm_continuous=resolution_config['resample_algorithm_continuous'],
                resample_algorithm_categorical=resolution_config['resample_algorithm_categorical']
            )
            
            logger.info(f"{input_data_name}: Preprocessing tiles...")
            tiles_paths = tiles.preprocess_tiles(data_type=input_data_values['data_type'])                        
            logger.info(f"{input_data_name}: Generated {len(tiles_paths)} tiles!")

            for year in input_data_yearly_data_index.keys():
                input_data_yearly_data_index[year][input_data_name] = tiles_paths

        return input_data_yearly_data_index

    def get_tiles_names(self, input_data_yearly_data_index: dict) -> list:
        tiles_names = []
        for _, source_yearly_data in input_data_yearly_data_index.items():
            for _, input_data_data_paths in source_yearly_data.items():
                for input_data_data_path in input_data_data_paths:
                    tiles_names.append(input_data_data_path.stem)
        return tiles_names

    def stack_data(self, input_data_yearly_data_index: dict, tiles_names: list, dataset_folder_path: Path, periods_config: dict, resolution_config: dict, dynamic_input_data: list, static_input_data: list, target_years_ranges: list):
        logger.info("Stacking data...")
        
        for target_years_range in target_years_ranges:
            
            logger.info(f"Stacking data for target years range: [{target_years_range[0]}, {target_years_range[-1]}]...")
            
            stacked_input_data_folder_path = dataset_folder_path / Path("input_data") / Path(f"{target_years_range[0]}_{target_years_range[-1]}")
            
            stacked_input_data_folder_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Stacked input data folder path: {str(stacked_input_data_folder_path)}")
            
            for tile_name in tiles_names:
                                
                number_of_bands = 0
                
                dynamic_data_to_stack = []
                
                for dynamic_input_data_name, dynamic_input_data_values in dynamic_input_data:
                    is_affected_by_fires = dynamic_input_data_values['is_affected_by_fires']
                    dynamic_input_data_years_range = self.get_dynamic_input_data_years_range_for_target_years_range(target_years_range, periods_config, is_affected_by_fires)
                    number_of_bands += len(dynamic_input_data_years_range)
                    dynamic_data_to_stack.append((dynamic_input_data_name, dynamic_input_data_years_range, dynamic_input_data_values['layer']))
                
                for static_input_data_name, _ in static_input_data:
                    number_of_bands += 1

                x_size = resolution_config['tile_size_in_pixels']
                y_size = resolution_config['tile_size_in_pixels']
                data_type = gdal.GDT_Float32

                driver = gdal.GetDriverByName('netCDF')
                stacked_tile_data_output_path = stacked_input_data_folder_path / Path(f"{tile_name}.nc")
                stacked_tile_ds = driver.Create(stacked_tile_data_output_path, x_size, y_size, number_of_bands, data_type)
        
                band_index = 1
                
                stacked_ds_georeferenced = False
                
                for dynamic_input_data_name, dynamic_input_data_years_range, dynamic_input_data_layer in dynamic_data_to_stack:
                    for year in dynamic_input_data_years_range:
                        year_data_tile_paths = input_data_yearly_data_index[year][dynamic_input_data_name]
                        
                        output_band = stacked_tile_ds.GetRasterBand(band_index)
                        
                        for year_data_tile_path in year_data_tile_paths:
                            if year_data_tile_path.stem == tile_name:
                                file_path = f"NETCDF:\"{year_data_tile_path.resolve()}\"{':' + dynamic_input_data_layer if dynamic_input_data_layer != '' else ''}"
                                input_ds = gdal.Open(file_path)
                                
                                if not stacked_ds_georeferenced:
                                    stacked_tile_ds.SetGeoTransform(input_ds.GetGeoTransform())
                                    stacked_tile_ds.SetProjection(input_ds.GetProjection())
                                    stacked_ds_georeferenced = True
                                    
                                input_band = input_ds.GetRasterBand(1)
                                input_band_data = input_band.ReadAsArray()
                                output_band.SetDescription(f"{dynamic_input_data_name}_{year}")
                                input_band_no_data_value = input_band.GetNoDataValue()
                                input_band_data = self.no_data_value_preprocessor.replace_no_data_values(input_band_data, input_band_no_data_value)
                                output_band.WriteArray(input_band_data)
                                stacked_tile_ds.FlushCache()
                                break
                        
                        band_index += 1
                
                for static_input_data_name, static_input_data_values in static_input_data:
                    output_band = stacked_tile_ds.GetRasterBand(band_index)
                    static_data_tile_paths = input_data_yearly_data_index[target_years_range[0]][static_input_data_name]
                    layer = static_input_data_values['layer']
                    
                    for tile_path in static_data_tile_paths:
                        if tile_path.stem == tile_name:
                            file_path = f"NETCDF:\"{tile_path.resolve()}\"{':' + layer if layer != '' else ''}"
                            input_ds = gdal.Open(file_path, gdal.GA_ReadOnly)
                            
                            if not stacked_ds_georeferenced:
                                    stacked_tile_ds.SetGeoTransform(input_ds.GetGeoTransform())
                                    stacked_tile_ds.SetProjection(input_ds.GetProjection())
                                    stacked_ds_georeferenced = True
                                    
                            input_band = input_ds.GetRasterBand(1)
                            input_band_data = input_band.ReadAsArray()
                            output_band.SetDescription(f"{static_input_data_name}")
                            input_band_no_data_value = input_band.GetNoDataValue()
                            input_band_data = self.no_data_value_preprocessor.replace_no_data_values(input_band_data, input_band_no_data_value)    
                            output_band.WriteArray(input_band_data)
                            stacked_tile_ds.FlushCache()
                            break
                    
                    band_index += 1    

    def get_dynamic_input_data_years_range_for_target_years_range(self, target_years_range: range, periods_config: dict, is_affected_by_fires: bool) -> range:
        if is_affected_by_fires:
            year_end_inclusive = target_years_range[0] - 1
            year_start_inclusive = year_end_inclusive - periods_config['input_data_affected_by_fires_period_length_in_years'] + 1
        else:
            year_end_inclusive = target_years_range[-1]
            year_start_inclusive = target_years_range[0]
        
        return range(year_start_inclusive, year_end_inclusive+1)

    def log_input_data_processing_results(self, dataset_folder_path: Path, sources_yearly_data_index: dict):
        if self.debug:
            logs_folder = dataset_folder_path / Path("logs")
            logs_folder.mkdir(parents=True, exist_ok=True)
            
            logger.info("Saving sources yearly data index logs for debugging...")
            
            serializable_sources_yearly_data_index = {year: {source_name: [str(tile_path) for tile_path in tile_paths] for source_name, tile_paths in source_yearly_data.items()} for year, source_yearly_data in sources_yearly_data_index.items()}
            with open(logs_folder / "sources_yearly_data_index.json", "w") as f:
                json.dump(serializable_sources_yearly_data_index, f, indent=4)
    
    def generate_targets(self, dataset_folder_path: Path, tmp_path: Path, target_years_ranges: list, big_tiles_boundaries: gpd.GeoDataFrame, resolution_config: dict, projections_config: dict):
        logger.info("Generating targets...")
        
        fire_data_source = NbacFireDataSource(Path(self.input_folder_path))
        
        tmp_target_folder_path = tmp_path / Path("targets")
        
        max_nb_processes = max(1, (len(os.sched_getaffinity(0)) - 1)//2)
        
        logger.info(f"Max nb processes: {max_nb_processes}")

        target = FireOccurrenceTarget(
            fire_data_source=fire_data_source,
            boundary=self.canada_boundary,
            target_pixel_size_in_meters=resolution_config['pixel_size_in_meters'],
            target_srid=projections_config['target_srs'],
            output_folder_path=tmp_target_folder_path,
            max_nb_processes=max_nb_processes
        )
        
        target_ranges_combined_raster = target.generate_target_for_years_ranges(target_years_ranges)
        
        logger.info("Generating tiles for targets...")

        dataset_targets_output_folder_path = dataset_folder_path / Path("target")
        dataset_targets_output_folder_path.mkdir(parents=True, exist_ok=True)

        for years_range, combined_raster_path in target_ranges_combined_raster.items():
            tiles_preprocessing_output_folder = tmp_target_folder_path / f"{years_range[0]}_{years_range[-1]}/"
            
            tiles_preprocessor = TilesPreprocessor(
                raw_tiles_folder=combined_raster_path.parent,
                layer_name="",
                tile_size_in_pixels=resolution_config['tile_size_in_pixels'],
                pixel_size_in_meters=resolution_config['pixel_size_in_meters'],
                output_folder=tiles_preprocessing_output_folder,
                big_tiles_boundaries=big_tiles_boundaries,
                source_srs=projections_config['target_srs'],
                target_srs=projections_config['target_srs'],
                resample_algorithm_continuous=resolution_config['resample_algorithm_continuous'],
                resample_algorithm_categorical=resolution_config['resample_algorithm_categorical']
            )

            logger.info(f"Preprocessing tiles for target years range: {years_range}...")
            tiles_paths = tiles_preprocessor.preprocess_tiles(data_type="categorical")
            logger.info(f"Generated {len(tiles_paths)} tiles for target years range: {years_range}!")
            
            tiles_folder_path = tiles_paths[0].parent
            
            years_range_output_folder_path = dataset_targets_output_folder_path / Path(f"{years_range[0]}_{years_range[-1]}")
            years_range_output_folder_path.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Moving preprocessed tiles to final destination for target years range: {years_range}...")
            shutil.move(tiles_folder_path, years_range_output_folder_path)
        
        logger.info("Generating targets DONE !")
        if self.debug:
            logs_folder = dataset_folder_path / Path("logs")
            logs_folder.mkdir(parents=True, exist_ok=True)
            logger.info("Saving target ranges combined raster logs for debugging...")
            serializable_target_ranges_combined_raster = {str(years_range): str(combined_raster_path) for years_range, combined_raster_path in target_ranges_combined_raster.items()}
            with open(logs_folder / "target_ranges_combined_raster.json", "w") as f:
                json.dump(serializable_target_ranges_combined_raster, f, indent=4)
    
    def cleanup_tmp_folder(self, tmp_folder_path: Path):
        if not self.debug:
            logger.info("Cleaning up tmp folder...")
            shutil.rmtree(tmp_folder_path)
        else:
            logger.info("Not cleaning up tmp folder since debug mode enabled!")
    
    def cleanup_dataset_folder(self, dataset_folder_path: Path):
        if not self.debug:
            logger.info("Cleaning up dataset folder...")
            shutil.rmtree(dataset_folder_path)
        else:
            logger.info("Not cleaning up dataset folder since debug mode enabled!")
