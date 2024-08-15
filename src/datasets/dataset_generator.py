import geopandas as gpd
import uuid
import subprocess
import numpy as np
import multiprocessing as mp
from osgeo import gdal
from pathlib import Path
from loguru import logger
from boundaries.canada_boundary import CanadaBoundary
from grid.square_meters_grid import SquareMetersGrid
from data_sources.nbac_fire_data_source import NbacFireDataSource
from targets.fire_occurrence_target import FireOccurrenceTarget
from raster_io.read import get_extension, get_formatted_file_path


class DatasetGenerator:
    def __init__(
        self, canada_boundary: CanadaBoundary, grid: SquareMetersGrid, config: dict
    ):
        self.canada_boundary = canada_boundary
        self.grid = grid
        self.input_folder_path = Path(config["paths"]["input_folder_path"])
        self.output_folder_path = Path(config["paths"]["output_folder_path"])
        self.debug = config["debug"]
        self.max_io_concurrency = config["max_io_concurrency"]
        self.max_cpu_concurrency = config["max_cpu_concurrency"]
        self.config = config
        gdal.UseExceptions()

    async def generate_dataset(self):
        logger.info("Generating dataset...")

        dataset_folder_path = self.create_dataset_folder_path()

        try:
            tmp_folder_path = self.create_dataset_tmp_folder_path(dataset_folder_path)

            logger.info("Loading boundaries...")
            self.canada_boundary.load(provinces=self.config["boundaries"]["provinces"])
            self.canada_boundary.save(tmp_folder_path)

            logger.info("Generating big tiles boundaries...")
            big_tiles_boundaries = self.grid.get_tiles_boundaries(
                self.canada_boundary.boundary
            )
            logger.info(f"Generated {len(big_tiles_boundaries)} big tiles boundaries!")

            target_years_ranges = self.generate_target_years_ranges()

            self.process_input_data(
                dataset_folder_path,
                tmp_folder_path,
                big_tiles_boundaries,
                target_years_ranges,
            )

            await self.generate_targets(
                dataset_folder_path,
                tmp_folder_path,
                big_tiles_boundaries,
                target_years_ranges,
            )

        except BaseException as e:
            logger.error(f"Error: {e}")
            raise e

    async def generate_targets(
        self,
        dataset_folder_path: Path,
        tmp_path: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        target_years_ranges: list,
    ):
        logger.info("Generating targets...")

        fire_data_source = NbacFireDataSource(Path(self.input_folder_path))

        tmp_target_folder_path = tmp_path / Path("target")

        target = FireOccurrenceTarget(
            fire_data_source=fire_data_source,
            boundary=self.canada_boundary,
            target_pixel_size_in_meters=self.config["resolution"][
                "pixel_size_in_meters"
            ],
            target_srid=self.config["projections"]["target_srid"],
            output_folder_path=tmp_target_folder_path,
            output_format="GTiff",
            max_io_concurrency=self.max_io_concurrency,
            max_cpu_concurrency=self.max_cpu_concurrency,
        )

        target_ranges_combined_raster = await target.generate_target_for_years_ranges(
            target_years_ranges
        )

        dataset_targets_output_folder_path = dataset_folder_path / Path("target")
        dataset_targets_output_folder_path.mkdir(parents=True, exist_ok=True)

        for years_range, combined_raster_path in target_ranges_combined_raster.items():
            with logger.contextualize(years_range=str(years_range)):
                tiles_preprocessing_output_folder = (
                    tmp_target_folder_path
                    / f"tiles_{years_range[0]}_{years_range[-1]}/"
                )

                logger.info("Resizing and reprojecting...")
                data_type = "categorical"
                resized_file_path = self.resize_and_reproject(
                    combined_raster_path,
                    tiles_preprocessing_output_folder,
                    data_type,
                    source_srid=self.config["projections"]["target_srid"],
                )

                logger.info("Creating tiles...")
                years_range_output_folder_path = (
                    dataset_targets_output_folder_path
                    / Path(f"{years_range[0]}_{years_range[-1]}")
                )
                years_range_output_folder_path.mkdir(parents=True, exist_ok=True)
                self.create_tiles(
                    resized_file_path,
                    years_range_output_folder_path,
                    big_tiles_boundaries,
                    create_child_folder=False,
                )

        logger.info("Generating targets DONE !")

    def create_dataset_folder_path(self) -> Path:
        dataset_uuid = self.get_dataset_uuid()

        dataset_folder_path = self.output_folder_path / Path(dataset_uuid)
        dataset_folder_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset folder path : {str(dataset_folder_path)}")

        return dataset_folder_path

    def get_dataset_uuid(self) -> str:
        ds_uuid = str(uuid.uuid4())
        logger.info(f"Dataset UUID: {ds_uuid}")
        return ds_uuid

    def generate_target_years_ranges(self) -> list:
        target_year_start_inclusive = self.config["periods"][
            "target_year_start_inclusive"
        ]
        target_year_end_inclusive = self.config["periods"]["target_year_end_inclusive"]
        target_period_length_in_years = self.config["periods"][
            "target_period_length_in_years"
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

        logger.info(f"Target years ranges: {target_years_ranges}")

        return target_years_ranges

    def create_dataset_tmp_folder_path(self, dataset_folder_path: Path) -> Path:
        tmp_folder_path = dataset_folder_path / Path("tmp")
        tmp_folder_path.mkdir(parents=True, exist_ok=True)
        return tmp_folder_path

    def process_input_data(
        self,
        dataset_folder_path: Path,
        tmp_folder_path: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        target_years_ranges: list,
    ):
        input_data_index = self.process_dynamic_input_data(
            tmp_folder_path, big_tiles_boundaries
        )
        self.process_static_input_data(
            tmp_folder_path, big_tiles_boundaries, input_data_index
        )
        self.stack_input_data(
            dataset_folder_path,
            target_years_ranges,
            input_data_index,
        )
        return input_data_index

    def stack_input_data(
        self,
        output_folder_path: Path,
        target_years_ranges: list,
        input_data_index: dict,
    ):
        with mp.Pool(processes=self.max_cpu_concurrency) as pool:
            args = [
                (output_folder_path, target_years_range, input_data_index)
                for target_years_range in target_years_ranges
            ]
            pool.starmap(self.stack_input_data_for_years_range, args)

    def stack_input_data_for_years_range(
        self,
        output_folder_path: Path,
        target_years_range: range,
        input_data_index: dict,
    ):
        logger.info(f"Stacking data for target years range {target_years_range}...")
        target_years_range_output_folder = (
            output_folder_path
            / Path("input_data")
            / Path(f"{target_years_range[0]}_{target_years_range[-1]}")
        )
        target_years_range_output_folder.mkdir(parents=True, exist_ok=True)

        tiles = [
            file_path.stem
            for file_path in list(list(input_data_index.values())[0].values())[0]
        ]

        for tile_name in tiles:

            files_to_stack = []

            for data_name, years_index in input_data_index.items():
                is_affected_by_fires = self.get_is_affected_by_fires(data_name)

                data_years_range = (
                    self.get_dynamic_input_data_years_range_for_target_years_range(
                        target_years_range, is_affected_by_fires
                    )
                )

                for year in data_years_range:
                    data_year_tiles_paths = list(years_index[year])
                    data_year_tile_path = next(
                        filter(
                            lambda path: path.stem == tile_name, data_year_tiles_paths
                        )
                    )
                    files_to_stack.append(str(data_year_tile_path))

            self.stack_files(
                files_to_stack, target_years_range_output_folder, tile_name
            )

    def stack_files(self, files_to_stack: list, output_folder: Path, tile_name: str):
        output_file = output_folder / f"{tile_name}{get_extension('gtiff')}"

        vrt_file = output_file.with_suffix(".vrt")

        gdalbuildvrt_cmd = f"gdalbuildvrt -separate {str(vrt_file)} " + " ".join(
            files_to_stack
        )
        self.run_command(gdalbuildvrt_cmd)

        gdal_translate_cmd = f"gdal_translate -strict -ot Float32 -of GTiff {str(vrt_file)} {str(output_file)}"
        self.run_command(gdal_translate_cmd)

        vrt_file.unlink()

    def get_is_affected_by_fires(self, data_name: str) -> bool:
        dynamic_data = self.config["input_data"].get("dynamic", {})
        for data_name, data_info in dynamic_data.items():
            if data_name == data_name:
                return data_info.get("is_affected_by_fires", False)

        static_data = self.config["input_data"].get("static", {})
        for data_name, data_info in static_data.items():
            if data_name == data_name:
                return False

        raise ValueError(f"Data name {data_name} not found in input data!")

    def get_dynamic_input_data_years_range_for_target_years_range(
        self,
        target_years_range: range,
        is_affected_by_fires: bool,
    ) -> range:
        if is_affected_by_fires:
            year_end_inclusive = target_years_range[0] - 1
            year_start_inclusive = (
                year_end_inclusive
                - self.config["periods"][
                    "input_data_affected_by_fires_period_length_in_years"
                ]
                + 1
            )
        else:
            year_start_inclusive = target_years_range[0]
            year_end_inclusive = target_years_range[-1]

        return range(year_start_inclusive, year_end_inclusive + 1)

    def process_dynamic_input_data(
        self, output_folder_path: Path, big_tiles_boundaries: gpd.GeoDataFrame
    ) -> dict:
        dynamic_data = self.config["input_data"].get("dynamic", {})
        args = [
            (output_folder_path, data_name, data_info, big_tiles_boundaries)
            for data_name, data_info in dynamic_data.items()
        ]

        with mp.Pool(processes=self.max_cpu_concurrency) as pool:
            input_data_years_indexes = pool.starmap(
                self.process_dynamic_input_data_years, args
            )

        input_data_index = {}
        for input_data_years_index in input_data_years_indexes:
            for data_name, years_index in input_data_years_index.items():
                input_data_index[data_name] = years_index

        return input_data_index

    def process_dynamic_input_data_years(
        self,
        output_folder_path: Path,
        data_name: str,
        data_info: dict,
        big_tiles_boundaries: gpd.GeoDataFrame,
    ) -> dict:
        with logger.contextualize(data_name=data_name):
            is_affected_by_fires = data_info.get("is_affected_by_fires", False)

            data_years_index = {data_name: {}}

            for year in self.get_dynamic_input_data_total_years_extent(
                is_affected_by_fires
            ):
                with logger.contextualize(year=year):
                    months_files_paths = []
                    for month in self.get_months_range():
                        with logger.contextualize(month=month):
                            data_input_folder = (
                                self.input_folder_path
                                / Path(f"{year}")
                                / Path(f"{month}")
                                / Path(f"{data_name}")
                            )
                            month_output_folder = (
                                output_folder_path
                                / Path(f"{year}")
                                / Path(f"{month}")
                                / Path(f"{data_name}")
                            )

                            logger.info("Merging files...")
                            merged_file_path = self.merge_spatially(
                                data_input_folder,
                                month_output_folder,
                                data_info,
                            )

                            logger.info("Aggregating bands for month...")
                            month_aggregated_file_path = self.aggregate_file(
                                merged_file_path, month_output_folder, data_info
                            )
                            months_files_paths.append(month_aggregated_file_path)

                    year_output_folder = (
                        output_folder_path / Path(f"{year}") / data_name
                    )
                    year_output_folder.mkdir(parents=True, exist_ok=True)

                    logger.info("Aggregating data for year...")
                    yearly_aggregated_file_path = self.aggregate_files(
                        months_files_paths, year_output_folder, data_info
                    )

                    logger.info("Resizing and reprojecting...")
                    data_type = data_info.get("data_type", None)
                    resized_file_path = self.resize_and_reproject(
                        yearly_aggregated_file_path,
                        year_output_folder,
                        data_type,
                        source_srid=self.config["projections"]["source_srid"],
                    )

                    logger.info("Creating tiles...")
                    tiles_files_paths = self.create_tiles(
                        resized_file_path, year_output_folder, big_tiles_boundaries
                    )
                    data_years_index[data_name][year] = tiles_files_paths

        return data_years_index

    def process_static_input_data(
        self,
        output_folder_path: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        input_data_index: dict,
    ):
        static_data = self.config["input_data"].get("static", {})
        args = [
            (output_folder_path, data_name, data_info, big_tiles_boundaries)
            for data_name, data_info in static_data.items()
        ]

        with mp.Pool(processes=self.max_cpu_concurrency) as pool:
            inputs_data_files_paths = pool.starmap(
                self.process_static_input_data_layer, args
            )

        all_years = set()
        for year_data in input_data_index.values():
            for year in year_data.keys():
                all_years.add(year)

        for input_data_files_paths in inputs_data_files_paths:
            for data_name, files_paths in input_data_files_paths.items():
                input_data_index[data_name] = {}
                for year in all_years:
                    input_data_index[data_name][year] = files_paths

    def process_static_input_data_layer(
        self,
        output_folder_path: Path,
        data_name: str,
        data_info: dict,
        big_tiles_boundaries: gpd.GeoDataFrame,
    ):
        with logger.contextualize(data_name=data_name):

            static_input_data_folder_name = "static_data"

            input_folder_path = (
                Path(self.input_folder_path)
                / Path(static_input_data_folder_name)
                / Path(f"{data_name}")
            )

            static_output_folder_path = (
                output_folder_path
                / Path(static_input_data_folder_name)
                / Path(data_name)
            )
            static_output_folder_path.mkdir(parents=True, exist_ok=True)

            logger.info("Merging files...")
            merged_file_path = self.merge_spatially(
                input_folder_path,
                static_output_folder_path,
                data_info,
            )

            logger.info("Resizing and reprojecting...")
            data_type = "categorical"
            resized_file_path = self.resize_and_reproject(
                merged_file_path,
                static_output_folder_path,
                data_type,
                source_srid=self.config["projections"]["source_srid"],
            )

            logger.info("Creating tiles...")
            tiles_files_paths = self.create_tiles(
                resized_file_path, static_output_folder_path, big_tiles_boundaries
            )

        return {data_name: tiles_files_paths}

    def create_tiles(
        self,
        input_file: Path,
        output_folder_path: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        create_child_folder: bool = True,
    ) -> list:
        if create_child_folder:
            tiles_output_folder = output_folder_path / "tiles"
            tiles_output_folder.mkdir(parents=True, exist_ok=True)
        else:
            tiles_output_folder = output_folder_path

        tiles_paths = []

        for tile_index, tile in big_tiles_boundaries.iterrows():
            tile_output_file = (
                tiles_output_folder / f"tile_{tile_index}{get_extension('gtiff')}"
            )
            bounds = tile["geometry"].bounds
            minx, miny, maxx, maxy = bounds
            self.run_command(
                f"gdalwarp -multi -of GTiff -te {minx} {miny} {maxx} {maxy} {str(input_file)} {str(tile_output_file)}"
            )
            tiles_paths.append(tile_output_file)

        return tiles_paths

    def resize_and_reproject(
        self,
        input_file: Path,
        output_folder_path: Path,
        data_type: str,
        source_srid: int,
    ):
        output_file_path = (
            output_folder_path
            / Path("resized_reprojected")
            / Path(f"resized_reprojected{get_extension('gtiff')}")
        )
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        target_srid = self.config["projections"]["target_srid"]

        pixel_size_in_meters = self.config["resolution"]["pixel_size_in_meters"]

        resampling_algorithm = self.get_resampling_algorithm(data_type)

        self.run_command(
            f"gdalwarp -multi -r {resampling_algorithm} -s_srs EPSG:{source_srid} -t_srs EPSG:{target_srid} -tr {pixel_size_in_meters} {pixel_size_in_meters} -cutline_srs EPSG:{self.canada_boundary.target_epsg} -cutline {str(self.canada_boundary.boundary_file)} -crop_to_cutline -of GTiff {str(input_file)} {str(output_file_path)}"
        )

        return output_file_path

    def get_resampling_algorithm(self, data_type: str) -> str:
        if data_type == "continuous":
            resample_algorithm = self.config["resolution"][
                "resample_algorithm_continuous"
            ]
        elif data_type == "categorical":
            resample_algorithm = self.config["resolution"][
                "resample_algorithm_categorical"
            ]
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return resample_algorithm

    def aggregate_file(
        self, input_file: Path, output_folder_path: Path, data_info: dict
    ) -> Path:
        output_file_path = (
            output_folder_path
            / Path("aggregated")
            / Path(f"aggregated{get_extension('GTiff')}")
        )

        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if data_info.get("aggregate_by", None) == "average":
            operation = np.mean
        elif data_info.get("aggregate_by", None) == "max":
            operation = np.max
        else:
            raise ValueError(
                f"Unknown aggregation strategy {data_info.get('aggregate_by', None)}!"
            )

        input_dataset = gdal.Open(str(input_file), gdal.GA_ReadOnly)
        xsize = input_dataset.RasterXSize
        ysize = input_dataset.RasterYSize
        block_size = 2048

        output_driver = gdal.GetDriverByName("GTiff")
        output_dataset = output_driver.Create(
            str(output_file_path.resolve()),
            xsize=xsize,
            ysize=ysize,
            bands=1,
            eType=gdal.GDT_Float32,
        )
        output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())
        output_dataset.SetProjection(input_dataset.GetProjection())
        output_band = output_dataset.GetRasterBand(1)
        output_band.SetNoDataValue(input_dataset.GetRasterBand(1).GetNoDataValue())

        for y in range(0, ysize, block_size):
            for x in range(0, xsize, block_size):
                x_block_size = min(block_size, xsize - x)
                y_block_size = min(block_size, ysize - y)

                aggregated_data = np.zeros(
                    (y_block_size, x_block_size), dtype=np.float32
                )

                for i in range(1, input_dataset.RasterCount + 1):
                    band = input_dataset.GetRasterBand(i)
                    data = band.ReadAsArray(x, y, x_block_size, y_block_size)
                    aggregated_data = operation([aggregated_data, data], axis=0)

                output_band.WriteArray(aggregated_data, x, y)

        output_dataset.FlushCache()

        del input_dataset
        del output_dataset

        return output_file_path

    def aggregate_files(
        self, input_files: list, output_folder_path: Path, data_info: dict
    ) -> Path:
        output_file_path = (
            output_folder_path
            / Path("aggregated")
            / Path(f"aggregated{get_extension('GTiff')}")
        )

        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        if data_info.get("aggregate_by", None) == "average":
            operation = np.mean
        elif data_info.get("aggregate_by", None) == "max":
            operation = np.max
        else:
            raise ValueError(
                f"Unknown aggregation strategy {data_info.get('aggregate_by', None)}!"
            )

        first_dataset = gdal.Open(str(input_files[0]), gdal.GA_ReadOnly)
        xsize = first_dataset.RasterXSize
        ysize = first_dataset.RasterYSize
        block_size = 2048

        output_driver = gdal.GetDriverByName("GTiff")
        output_dataset = output_driver.Create(
            str(output_file_path.resolve()),
            xsize=xsize,
            ysize=ysize,
            bands=1,
            eType=gdal.GDT_Float32,
        )
        output_dataset.SetGeoTransform(first_dataset.GetGeoTransform())
        output_dataset.SetProjection(first_dataset.GetProjection())
        output_band = output_dataset.GetRasterBand(1)
        output_band.SetNoDataValue(first_dataset.GetRasterBand(1).GetNoDataValue())

        for y in range(0, ysize, block_size):
            for x in range(0, xsize, block_size):
                x_block_size = min(block_size, xsize - x)
                y_block_size = min(block_size, ysize - y)

                aggregated_data = np.zeros(
                    (y_block_size, x_block_size), dtype=np.float32
                )

                for file in input_files:
                    dataset = gdal.Open(str(file), gdal.GA_ReadOnly)
                    band = dataset.GetRasterBand(1)
                    data = band.ReadAsArray(x, y, x_block_size, y_block_size)
                    aggregated_data = operation([aggregated_data, data], axis=0)
                    del dataset

                output_band.WriteArray(aggregated_data, x, y)

        output_dataset.FlushCache()

        del first_dataset
        del output_dataset

        return output_file_path

    def merge_spatially(
        self, input_folder: Path, month_output_folder: Path, data_info: dict
    ) -> Path:
        netcdf_layer = data_info.get("netcdf_layer", None)

        input_files = " ".join(
            [
                get_formatted_file_path(file, netcdf_layer)
                for file in input_folder.glob(f"*{get_extension('netcdf')}")
            ]
        )
        output_file = (
            month_output_folder / Path("merged") / f"merged{get_extension('gtiff')}"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.run_command(
            f"gdalwarp -multi -of GTiff -overwrite {input_files} {str(output_file)}"
        )

        return output_file

    def get_months_range(self) -> range:
        return range(
            self.config["periods"]["month_start_inclusive"],
            self.config["periods"]["month_end_inclusive"] + 1,
        )

    def get_dynamic_input_data_total_years_extent(
        self, is_affected_by_fires: bool
    ) -> range:
        if is_affected_by_fires:
            year_start_inclusive = (
                self.config["periods"]["target_year_start_inclusive"]
                - self.config["periods"][
                    "input_data_affected_by_fires_period_length_in_years"
                ]
            )
            year_end_inclusive = (
                self.config["periods"]["target_year_end_inclusive"]
                - self.config["periods"]["target_period_length_in_years"]
            )
        else:
            year_start_inclusive = self.config["periods"]["target_year_start_inclusive"]
            year_end_inclusive = self.config["periods"]["target_year_end_inclusive"]

        return range(year_start_inclusive, year_end_inclusive + 1)

    def run_command(self, command: str):
        try:
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            raise e
