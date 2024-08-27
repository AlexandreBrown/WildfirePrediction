import shutil
import asyncio
import geopandas as gpd
import uuid
import time
import numpy as np
from osgeo import gdal
from pathlib import Path
from loguru import logger
from typing import Optional
from boundaries.canada_boundary import CanadaBoundary
from grid.square_meters_grid import SquareMetersGrid
from data_sources.nbac_fire_data_source import NbacFireDataSource
from targets.fire_occurrence_target import FireOccurrenceTarget
from raster_io.read import get_extension, get_formatted_file_path
from logging_utils.logging import get_ram_used, get_ram_total


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

        logger.opt(lazy=True).debug(
            "RAM Usage: {used:.2f}/{total:.2f}",
            used=get_ram_used(),
            total=get_ram_total(),
        )

        try:
            dataset_folder_path = self.get_dataset_folder_path()
            tmp_folder_path = self.get_dataset_tmp_folder_path(dataset_folder_path)

            logger.info("Loading boundaries...")
            self.canada_boundary.load(provinces=self.config["boundaries"]["provinces"])
            self.canada_boundary.save(tmp_folder_path)

            logger.info("Generating big tiles boundaries...")
            big_tiles_boundaries = self.grid.get_tiles_boundaries(
                self.canada_boundary.boundary
            )
            logger.info(f"Generated {len(big_tiles_boundaries)} big tiles boundaries!")

            target_years_ranges = self.generate_target_years_ranges()

            await self.process_input_data(
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

            logger.opt(lazy=True).debug(
                "RAM Usage: {used:.2f}/{total:.2f}",
                used=get_ram_used(),
                total=get_ram_total(),
            )

            if self.config["cleanup_tmp_folder_on_success"]:
                self.cleanup_folder(tmp_folder_path)

            logger.success("Dataset generation DONE !")
        except BaseException as e:
            logger.error(f"Error: {e}")
            raise e

    def get_dataset_folder_path(self) -> Path:
        if self.config["resume"]:
            logger.info("Resuming dataset generation...")
            dataset_folder_path = Path(self.config["resume_folder_path"])
        else:
            dataset_folder_path = self.create_dataset_folder_path()

        logger.info(f"Dataset folder path: {dataset_folder_path}")

        return dataset_folder_path

    def get_dataset_tmp_folder_path(self, dataset_folder_path: Path) -> Path:
        if self.config["resume"]:
            tmp_folder_path = dataset_folder_path / Path("tmp")
        else:
            tmp_folder_path = self.create_dataset_tmp_folder_path(dataset_folder_path)

        return tmp_folder_path

    def cleanup_folder(self, folder_path: Path):
        if self.debug:
            return
        logger.info(f"Cleaning up folder {folder_path}...")
        shutil.rmtree(folder_path)

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

                logger.info("Resizing, reprojecting and clipping raster...")
                data_type = "categorical"
                clipped_file_path = await self.resize_reproject_clip_raster(
                    combined_raster_path,
                    tiles_preprocessing_output_folder,
                    data_type,
                    source_srid=self.config["projections"]["target_srid"],
                    out_type="Int8",
                    out_no_data_value=int(
                        self.config["target_new_no_data_value"],
                    ),
                )

                logger.info("Creating tiles...")
                years_range_output_folder_path = (
                    dataset_targets_output_folder_path
                    / Path(f"{years_range[0]}_{years_range[-1]}")
                )
                years_range_output_folder_path.mkdir(parents=True, exist_ok=True)
                await self.create_tiles(
                    clipped_file_path,
                    years_range_output_folder_path,
                    big_tiles_boundaries,
                    create_child_folder=False,
                )
                self.cleanup_file(clipped_file_path)

        logger.info("Generating targets DONE !")

    def create_dataset_folder_path(self) -> Path:
        dataset_uuid = self.get_dataset_uuid()

        dataset_folder_path = self.output_folder_path / Path(dataset_uuid)
        dataset_folder_path.mkdir(parents=True, exist_ok=True)

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

    async def process_input_data(
        self,
        dataset_folder_path: Path,
        tmp_folder_path: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        target_years_ranges: list,
    ):
        data_index = {}

        semaphore = asyncio.Semaphore(self.max_io_concurrency)

        if not self.config["resume"]:

            dynamic_data = self.config["input_data"].get("dynamic", {})
            dynamic_input_data_args = [
                (
                    self.process_dynamic_input_data,
                    tmp_folder_path,
                    data_name,
                    data_info,
                    big_tiles_boundaries,
                    semaphore,
                )
                for data_name, data_info in dynamic_data.items()
            ]

            static_data = self.config["input_data"].get("static", {})
            static_input_data_args = [
                (
                    self.process_static_input_data,
                    tmp_folder_path,
                    data_name,
                    data_info,
                    big_tiles_boundaries,
                    semaphore,
                )
                for data_name, data_info in static_data.items()
            ]

            args = dynamic_input_data_args + static_input_data_args

            tasks = set()

            for arg in args:
                task = asyncio.create_task(self._process_input_data(*arg))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            input_data_indexes = await asyncio.gather(*tasks)

            for input_data_index in input_data_indexes:
                for data_name, years_index_or_files_paths in input_data_index.items():
                    data_index[data_name] = years_index_or_files_paths
        else:
            logger.info("Resuming input data processing...")
            data_index = self.get_resumed_input_data_index(tmp_folder_path)
            logger.info("Loaded data index from already processed data!")

        await self.stack_input_data(
            dataset_folder_path, target_years_ranges, data_index, semaphore
        )

    def get_resumed_input_data_index(self, tmp_folder_path: Path) -> dict:
        resumed_input_data_index = {}

        dynamic_data = self.config["input_data"].get("dynamic", {})
        for data_name, data_info in dynamic_data.items():
            resumed_input_data_index[data_name] = (
                self.get_resumed_input_data_index_for_dynamic_input_data(
                    tmp_folder_path, data_name, data_info
                )
            )

        static_data = self.config["input_data"].get("static", {})
        for data_name, data_info in static_data.items():
            resumed_input_data_index[data_name] = (
                self.get_resumed_input_data_index_for_static_input_data(
                    tmp_folder_path, data_name
                )
            )

        return resumed_input_data_index

    def get_resumed_input_data_index_for_dynamic_input_data(
        self, tmp_folder_path: Path, data_name: str, data_info: dict
    ) -> dict:
        data_years_index = {}
        is_affected_by_fires = data_info.get("is_affected_by_fires", False)
        for year in self.get_dynamic_input_data_total_years_extent(
            is_affected_by_fires
        ):
            year_tiles_paths = list(
                (
                    tmp_folder_path / Path(str(year)) / Path(data_name) / Path("tiles")
                ).glob(f"*{get_extension('gtiff')}")
            )
            logger.info(
                f"Found {len(year_tiles_paths)} tiles for dynamic input data {data_name} and year {year}!"
            )
            data_years_index[year] = year_tiles_paths
        return data_years_index

    def get_resumed_input_data_index_for_static_input_data(
        self, tmp_folder_path: Path, data_name: str
    ) -> dict:
        tiles_paths = list(
            (
                tmp_folder_path / Path("static_data") / Path(data_name) / Path("tiles")
            ).glob(f"*{get_extension('gtiff')}")
        )
        logger.info(
            f"Found {len(tiles_paths)} tiles for static input data {data_name}!"
        )
        return tiles_paths

    async def _process_input_data(self, *args):
        processing_fn = args[0]
        return await processing_fn(*args[1:])

    async def process_dynamic_input_data(
        self,
        output_folder_path: Path,
        data_name: str,
        data_info: dict,
        big_tiles_boundaries: gpd.GeoDataFrame,
        semaphore: asyncio.Semaphore,
    ) -> dict:
        with logger.contextualize(data_name=data_name):
            is_affected_by_fires = data_info.get("is_affected_by_fires", False)

            tasks = set()
            for year in self.get_dynamic_input_data_total_years_extent(
                is_affected_by_fires
            ):
                task = asyncio.create_task(
                    self.process_dynamic_input_data_year(
                        output_folder_path,
                        data_name,
                        data_info,
                        big_tiles_boundaries,
                        year,
                        semaphore,
                    )
                )
                tasks.add(task)
                task.add_done_callback(tasks.discard)

            results = await asyncio.gather(*tasks)

            data_years_index = {data_name: {}}

            for year, tiles_files_paths in results:
                data_years_index[data_name][year] = tiles_files_paths

            await logger.complete()

        return data_years_index

    async def process_dynamic_input_data_year(
        self,
        output_folder_path: Path,
        data_name: str,
        data_info: dict,
        big_tiles_boundaries: gpd.GeoDataFrame,
        year: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple:
        async with semaphore:
            with logger.contextualize(year=year):
                months_files_paths = []
                for month in self.get_months_range():
                    month_files_paths = await self.process_dynamic_input_data_month(
                        output_folder_path,
                        data_name,
                        data_info,
                        year,
                        month,
                    )
                    months_files_paths.append(month_files_paths)

                year_output_folder = output_folder_path / Path(f"{year}") / data_name
                year_output_folder.mkdir(parents=True, exist_ok=True)

                logger.info("Aggregating data for year...")
                yearly_aggregated_file_path = await asyncio.to_thread(
                    self.aggregate_files,
                    months_files_paths,
                    year_output_folder,
                    data_info,
                )
                logger.opt(lazy=True).debug(
                    "RAM Usage: {used:.2f}/{total:.2f}",
                    used=get_ram_used(),
                    total=get_ram_total(),
                )
                self.cleanup_files(months_files_paths)

                logger.info("Resizing, reprojecting and clipping raster...")
                data_type = data_info.get("data_type", None)
                clipped_file_path = await self.resize_reproject_clip_raster(
                    yearly_aggregated_file_path,
                    year_output_folder,
                    data_type,
                    source_srid=self.config["projections"]["source_srid"],
                )
                logger.opt(lazy=True).debug(
                    "RAM Usage: {used:.2f}/{total:.2f}",
                    used=get_ram_used(),
                    total=get_ram_total(),
                )
                self.cleanup_file(yearly_aggregated_file_path)

                logger.info("Creating tiles...")
                tiles_files_paths = await self.create_tiles(
                    clipped_file_path, year_output_folder, big_tiles_boundaries
                )
                self.cleanup_file(clipped_file_path)

                return year, tiles_files_paths

    async def process_dynamic_input_data_month(
        self,
        output_folder_path: Path,
        data_name: str,
        data_info: dict,
        year: int,
        month: int,
    ):
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
            logger.opt(lazy=True).debug(
                "RAM Usage: {used:.2f}/{total:.2f}",
                used=get_ram_used(),
                total=get_ram_total(),
            )
            merged_file_path = await self.merge_spatially(
                data_input_folder,
                month_output_folder,
                data_info,
            )
            logger.opt(lazy=True).debug(
                "RAM Usage: {used:.2f}/{total:.2f}",
                used=get_ram_used(),
                total=get_ram_total(),
            )

            logger.info("Aggregating bands for month...")
            month_aggregated_file_path = await asyncio.to_thread(
                self.aggregate_file,
                merged_file_path,
                month_output_folder,
                data_info,
            )
            logger.opt(lazy=True).debug(
                "RAM Usage: {used:.2f}/{total:.2f}",
                used=get_ram_used(),
                total=get_ram_total(),
            )
            self.cleanup_file(merged_file_path)
            return month_aggregated_file_path

    def cleanup_files(self, files: list):
        for file in files:
            self.cleanup_file(file)

    def cleanup_file(self, file: Path):
        if self.debug:
            return
        file.unlink()

    async def process_static_input_data(
        self,
        output_folder_path: Path,
        data_name: str,
        data_info: dict,
        big_tiles_boundaries: gpd.GeoDataFrame,
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
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
                merged_file_path = await self.merge_spatially(
                    input_folder_path,
                    static_output_folder_path,
                    data_info,
                )
                logger.opt(lazy=True).debug(
                    "RAM Usage: {used:.2f}/{total:.2f}",
                    used=get_ram_used(),
                    total=get_ram_total(),
                )

                logger.info("Resizing, reprojecting and clipping raster...")
                data_type = "categorical"
                clipped_file_path = await self.resize_reproject_clip_raster(
                    merged_file_path,
                    static_output_folder_path,
                    data_type,
                    source_srid=self.config["projections"]["source_srid"],
                )
                logger.opt(lazy=True).debug(
                    "RAM Usage: {used:.2f}/{total:.2f}",
                    used=get_ram_used(),
                    total=get_ram_total(),
                )
                self.cleanup_file(merged_file_path)

                logger.info("Creating tiles...")
                tiles_files_paths = await self.create_tiles(
                    clipped_file_path, static_output_folder_path, big_tiles_boundaries
                )
                self.cleanup_file(clipped_file_path)

            logger.opt(lazy=True).debug(
                "RAM Usage: {used:.2f}/{total:.2f}",
                used=get_ram_used(),
                total=get_ram_total(),
            )

            await logger.complete()

            return {data_name: tiles_files_paths}

    async def create_tiles(
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
            await self.run_command(
                "gdalwarp",
                [
                    "--quiet",
                    "-overwrite",
                    "-multi",
                    "-of",
                    "GTiff",
                    "-te",
                    str(minx),
                    str(miny),
                    str(maxx),
                    str(maxy),
                    str(input_file),
                    str(tile_output_file),
                ],
            )
            tiles_paths.append(tile_output_file)

        return tiles_paths

    async def resize_reproject_clip_raster(
        self,
        input_file: Path,
        output_folder_path: Path,
        data_type: str,
        source_srid: int,
        out_type: Optional[str] = None,
        out_no_data_value=None,
    ):
        target_srid = self.config["projections"]["target_srid"]

        pixel_size_in_meters = self.config["resolution"]["pixel_size_in_meters"]

        resampling_algorithm = self.get_resampling_algorithm(data_type)

        xmin, ymin, xmax, ymax = self.canada_boundary.boundary.total_bounds

        resized_output_file_path = (
            output_folder_path
            / Path("resized_reprojected")
            / Path(f"resized_reprojected{get_extension('gtiff')}")
        )
        resized_output_file_path.parent.mkdir(parents=True, exist_ok=True)
        await self.run_command(
            "gdalwarp",
            [
                "--quiet",
                "-overwrite",
                "-multi",
                "-te",
                str(xmin),
                str(ymin),
                str(xmax),
                str(ymax),
                "-r",
                str(resampling_algorithm),
                "-s_srs",
                f"EPSG:{source_srid}",
                "-t_srs",
                f"EPSG:{target_srid}",
                "-tr",
                str(pixel_size_in_meters),
                str(pixel_size_in_meters),
                "-of",
                "GTiff",
                str(input_file),
                str(resized_output_file_path),
            ],
        )

        clipped_ouput_file_path = (
            output_folder_path
            / Path("clipped")
            / Path(f"clipped{get_extension('gtiff')}")
        )
        clipped_ouput_file_path.parent.mkdir(parents=True, exist_ok=True)

        params = [
            "--quiet",
            "-overwrite",
            "-multi",
            "-cutline_srs",
            f"EPSG:{self.canada_boundary.target_epsg}",
            "-cutline",
            f"{str(self.canada_boundary.boundary_file)}",
            "-crop_to_cutline",
        ]
        if out_type is not None and out_no_data_value is not None:
            params.append("-ot")
            params.append(str(out_type))
            params.append("-dstnodata")
            params.append(str(out_no_data_value))

        params.append("-of")
        params.append("GTiff")
        params.append(str(resized_output_file_path))
        params.append(str(clipped_ouput_file_path))

        await self.run_command(
            "gdalwarp",
            params,
        )

        self.cleanup_file(resized_output_file_path)

        return clipped_ouput_file_path

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

        aggregation_strategy = data_info["aggregation_strategy"]
        fill_values = data_info["fill_values"]

        input_dataset = gdal.Open(str(input_file), gdal.GA_ReadOnly)
        xsize = input_dataset.RasterXSize
        ysize = input_dataset.RasterYSize
        block_size = 8192

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

                sum_data = np.zeros((y_block_size, x_block_size), dtype=np.float32)
                max_data = np.full(
                    (y_block_size, x_block_size), -np.inf, dtype=np.float32
                )
                valid_pixel_count = np.zeros(
                    (y_block_size, x_block_size), dtype=np.float32
                )

                for i in range(1, input_dataset.RasterCount + 1):
                    band = input_dataset.GetRasterBand(i)
                    data = band.ReadAsArray(x, y, x_block_size, y_block_size)

                    mask_values_to_ignore = np.isin(
                        data, list(set(fill_values + [band.GetNoDataValue()]))
                    )
                    mask_values_to_aggregate = np.logical_not(mask_values_to_ignore)
                    valid_pixel_count[mask_values_to_aggregate] += 1

                    if aggregation_strategy == "average":
                        sum_data[mask_values_to_aggregate] += data[
                            mask_values_to_aggregate
                        ]
                    elif aggregation_strategy == "max":
                        max_data[mask_values_to_aggregate] = np.maximum(
                            max_data[mask_values_to_aggregate],
                            data[mask_values_to_aggregate],
                        )

                aggregated_data = np.full_like(
                    valid_pixel_count, fill_value=output_band.GetNoDataValue()
                )
                if aggregation_strategy == "average":
                    aggregated_data[valid_pixel_count > 0] = (
                        sum_data[valid_pixel_count > 0]
                        / valid_pixel_count[valid_pixel_count > 0]
                    )
                elif aggregation_strategy == "max":
                    aggregated_data = max_data

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

        aggregation_strategy = data_info["aggregation_strategy"]
        fill_values = data_info["fill_values"]

        first_dataset = gdal.Open(str(input_files[0]), gdal.GA_ReadOnly)
        xsize = first_dataset.RasterXSize
        ysize = first_dataset.RasterYSize
        block_size = 8192

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

                sum_data = np.zeros((y_block_size, x_block_size), dtype=np.float32)
                max_data = np.full(
                    (y_block_size, x_block_size), -np.inf, dtype=np.float32
                )
                valid_pixel_count = np.zeros(
                    (y_block_size, x_block_size), dtype=np.float32
                )

                for file in input_files:
                    dataset = gdal.Open(str(file), gdal.GA_ReadOnly)
                    band = dataset.GetRasterBand(1)
                    data = band.ReadAsArray(x, y, x_block_size, y_block_size)

                    mask_values_to_ignore = np.isin(
                        data, list(set(fill_values + [band.GetNoDataValue()]))
                    )
                    mask_values_to_aggregate = np.logical_not(mask_values_to_ignore)
                    valid_pixel_count[mask_values_to_aggregate] += 1

                    if aggregation_strategy == "average":
                        sum_data[mask_values_to_aggregate] += data[
                            mask_values_to_aggregate
                        ]
                    elif aggregation_strategy == "max":
                        max_data[mask_values_to_aggregate] = np.maximum(
                            max_data[mask_values_to_aggregate],
                            data[mask_values_to_aggregate],
                        )

                    del dataset

                aggregated_data = np.full_like(
                    valid_pixel_count, fill_value=output_band.GetNoDataValue()
                )
                if aggregation_strategy == "average":
                    aggregated_data[valid_pixel_count > 0] = (
                        sum_data[valid_pixel_count > 0]
                        / valid_pixel_count[valid_pixel_count > 0]
                    )
                elif aggregation_strategy == "max":
                    aggregated_data = max_data

                output_band.WriteArray(aggregated_data, x, y)

        output_dataset.FlushCache()

        del first_dataset
        del output_dataset

        return output_file_path

    async def merge_spatially(
        self, input_folder: Path, month_output_folder: Path, data_info: dict
    ) -> Path:
        netcdf_layer = data_info.get("netcdf_layer", None)

        input_files = [
            get_formatted_file_path(file, netcdf_layer)
            for file in input_folder.glob(f"*{get_extension('netcdf')}")
        ]
        output_file = (
            month_output_folder / Path("merged") / f"merged{get_extension('gtiff')}"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)

        await self.run_command(
            "gdalwarp",
            [
                "--quiet",
                "-overwrite",
                "-multi",
                "-of",
                "GTiff",
                *input_files,
                str(output_file),
            ],
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

    async def stack_input_data(
        self,
        output_folder_path: Path,
        target_years_ranges: list,
        input_data_index: dict,
        semaphore: asyncio.Semaphore,
    ):
        args = [
            (output_folder_path, target_years_range, input_data_index, semaphore)
            for target_years_range in target_years_ranges
        ]
        tasks = set()
        for arg in args:
            task = asyncio.create_task(self.stack_input_data_for_years_range(*arg))
            tasks.add(task)
            task.add_done_callback(tasks.discard)

        await asyncio.gather(*tasks)

        if self.config["cleanup_tmp_folder_on_success"]:
            for data_name, years_index_or_files_paths in input_data_index.items():
                if isinstance(years_index_or_files_paths, dict):
                    for year, tiles_paths in years_index_or_files_paths.items():
                        self.cleanup_files(tiles_paths)
                else:
                    self.cleanup_files(years_index_or_files_paths)

    async def stack_input_data_for_years_range(
        self,
        output_folder_path: Path,
        target_years_range: range,
        input_data_index: dict,
        semaphore: asyncio.Semaphore,
    ):
        async with semaphore:
            logger.info(f"Stacking data for target years range {target_years_range}...")
            logger.opt(lazy=True).debug(
                "RAM Usage: {used:.2f}/{total:.2f}",
                used=get_ram_used(),
                total=get_ram_total(),
            )
            target_years_range_output_folder = (
                output_folder_path
                / Path("input_data")
                / Path(f"{target_years_range[0]}_{target_years_range[-1]}")
            )
            target_years_range_output_folder.mkdir(parents=True, exist_ok=True)

            first_data_values = list(input_data_index.values())[0]

            if isinstance(first_data_values, list):
                tiles = [file_path.stem for file_path in first_data_values]
            else:
                tiles = [
                    file_path.stem for file_path in list(first_data_values.values())[0]
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

                    fill_values = self.get_fill_values(data_name)

                    for year in data_years_range:
                        if isinstance(years_index, list):
                            data_tiles_paths = years_index
                        else:
                            data_tiles_paths = list(years_index[year])
                        matching_tile_path = next(
                            filter(
                                lambda path: path.stem == tile_name, data_tiles_paths
                            )
                        )
                        self.update_nodata_value(matching_tile_path, fill_values)
                        files_to_stack.append(str(matching_tile_path))

                await self.stack_files(
                    files_to_stack, target_years_range_output_folder, tile_name
                )

            await logger.complete()

    def get_dynamic_input_data_years_range_for_target_years_range(
        self,
        target_years_range: range,
        is_affected_by_fires: bool,
    ) -> range:
        if is_affected_by_fires:
            year_start_inclusive = (
                target_years_range[0]
                - self.config["periods"][
                    "input_data_affected_by_fires_period_length_in_years"
                ]
            )

            year_end_inclusive = target_years_range[0] - 1
        else:
            year_start_inclusive = target_years_range[0]
            year_end_inclusive = target_years_range[-1]

        return range(year_start_inclusive, year_end_inclusive + 1)

    def get_fill_values(self, data_name: str) -> list:
        dynamic_data = self.config["input_data"].get("dynamic", {})
        for name, data_info in dynamic_data.items():
            if data_name == name:
                return data_info.get("fill_values", [])

        static_data = self.config["input_data"].get("static", {})
        for name, data_info in static_data.items():
            if data_name == name:
                return data_info.get("fill_values", [])

        raise ValueError(f"Data name {data_name} not found in input data!")

    def update_nodata_value(self, file_path: Path, fill_values: list):
        dataset = gdal.Open(str(file_path), gdal.GA_Update)

        new_nodata_value = float(self.config["input_data_new_no_data_value"])

        for band_index in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(band_index)
            previous_nodata_value = band.GetNoDataValue()
            combined_nodata_value_to_update = list(
                set(fill_values + [previous_nodata_value])
            )

            band_data = band.ReadAsArray()

            nodata_mask = np.isin(band_data, list(combined_nodata_value_to_update))
            band_data[nodata_mask] = new_nodata_value
            band.WriteArray(band_data)

            band.SetNoDataValue(new_nodata_value)
            dataset.FlushCache()

        del dataset

    async def stack_files(
        self, files_to_stack: list, output_folder: Path, tile_name: str
    ):
        output_file = output_folder / f"{tile_name}{get_extension('gtiff')}"

        vrt_file = output_file.with_suffix(".vrt")

        await self.run_command(
            "gdalbuildvrt",
            [
                "-overwrite",
                "-separate",
                str(vrt_file),
                *(files_to_stack),
            ],
        )

        await self.run_command(
            "gdal_translate",
            [
                "-strict",
                "-ot",
                "Float32",
                "-of",
                "GTiff",
                str(vrt_file),
                str(output_file),
            ],
        )

        vrt_file.unlink()

    def get_is_affected_by_fires(self, data_name_to_lookup: str) -> bool:
        dynamic_data = self.config["input_data"].get("dynamic", {})
        for data_name, data_info in dynamic_data.items():
            if data_name_to_lookup == data_name:
                return data_info.get("is_affected_by_fires", False)

        static_data = self.config["input_data"].get("static", {})
        for data_name, data_info in static_data.items():
            if data_name_to_lookup == data_name:
                return False

        raise ValueError(f"Data name {data_name} not found in input data!")

    async def run_command(self, program: str, commands: list):
        tries = 1
        max_tries = 100
        for _ in range(max_tries):
            try:
                proc = await asyncio.create_subprocess_exec(
                    program,
                    *commands,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr = await proc.communicate()

                error = stderr.decode()
                if error != "":
                    logger.error(error)
                    raise Exception(error)

                return
            except Exception as e:
                logger.error(
                    f"({tries}/{max_tries}) Program {program} failed executing commands {commands[:5]}: {e}, retrying..."
                )
                tries += 1
                time.sleep(tries * 5)
        exception_message = f"Program {program} failed executing commands {commands} after {tries} tries!"
        logger.exception(exception_message)
        raise Exception(exception_message)
