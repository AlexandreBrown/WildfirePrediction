import uuid
import shutil
import geopandas as gpd
import json
import subprocess
from loguru import logger
from boundaries.canada_boundary import CanadaBoundary
from data_sources.nbac_fire_data_source import NbacFireDataSource
from grid.square_meters_grid import SquareMetersGrid
from preprocessing.no_data_value_preprocessor import NoDataValuePreprocessor
from preprocessing.tiles_preprocessor import TilesPreprocessor
from pathlib import Path
from targets.fire_occurrence_target import FireOccurrenceTarget


class DatasetGeneratorV2:
    def __init__(
        self,
        canada_boundary: CanadaBoundary,
        grid: SquareMetersGrid,
        input_folder_path: Path,
        output_folder_path: Path,
        debug: bool,
        no_data_value_preprocessor: NoDataValuePreprocessor,
        input_format: str,
        output_format: str,
        max_io_concurrency: int,
        max_cpu_concurrency: int,
    ):
        self.canada_boundary = canada_boundary
        self.grid = grid
        self.input_folder_path = input_folder_path
        self.output_folder_path = output_folder_path
        self.debug = debug
        self.no_data_value_preprocessor = no_data_value_preprocessor
        self.input_format = input_format
        self.output_format = output_format
        self.max_io_concurrency = max_io_concurrency
        self.max_cpu_concurrency = max_cpu_concurrency
        gdal.UseExceptions()

    def generate(
        self,
        dynamic_input_data: list,
        static_input_data: list,
        periods_config: dict,
        resolution_config: dict,
        projections_config: dict,
    ):
        logger.info("Generating dataset...")

        dataset_folder_path = self.get_dataset_folder_path()

        try:
            tmp_folder_path = self.get_dataset_tmp_folder_path(dataset_folder_path)
            processed_data_output_folder_path = tmp_folder_path / Path("processed_data")

            year = 2022
            month = 5
            input_data_name = "MOD44B_061_Percent_Tree_Cover"
            raw_tiles_folder = (
                self.input_folder_path
                / Path(f"{year}")
                / Path(f"{month}")
                / Path(f"{input_data_name}")
            )

            output_path = (
                processed_data_output_folder_path
                / Path(f"{month}")
                / Path(f"{input_data_name}")
            )
            output_path.mkdir(parents=True, exist_ok=True)

            # Step 1: Convert calendar attribute to "standard" using ncatted
            converted_folder = output_path / "converted"
            converted_folder.mkdir(parents=True, exist_ok=True)
            for file in raw_tiles_folder.glob("*.nc"):
                output_file = converted_folder / file.name
                self.run_command(
                    f"ncatted -O -a calendar,time,o,c,standard {str(file)} {str(output_file)}"
                )

            # Step 2: Spatially merge the files into a single raster with multiple time steps
            merged_file = output_path / Path(f"{input_data_name}_merged.nc")
            input_files = " ".join(
                [
                    f'NETCDF:"{file}":Percent_Tree_Cover'
                    for file in converted_folder.glob("*.nc")
                ]
            )
            self.run_command(
                f"gdalwarp -of netCDF -overwrite {input_files} {str(merged_file)}"
            )

            # Step 3: Compute the average across all time steps
            averaged_file = output_path / Path(f"{input_data_name}_averaged.nc")
            self.run_command(f"cdo timmean {str(merged_file)} {str(averaged_file)}")

            # Step 4: Resize to 250m pixel size and reproject to EPSG:3978 using subprocess with multithreading
            final_output_file = output_path / Path(f"{input_data_name}_final.nc")
            self.run_command(
                f"gdalwarp -multi -wo NUM_THREADS=ALL_CPUS -t_srs EPSG:3978 -tr 250 250 -of netCDF {str(averaged_file)} {str(final_output_file)}"
            )

            # Step 5: Generate 512x512 Pixel Tiles
            logger.info("Generating big tiles boundaries...")
            big_tiles_boundaries = self.grid.get_tiles_boundaries(
                self.canada_boundary.boundary
            )
            logger.info(f"Generated {len(big_tiles_boundaries)} big tiles boundaries!")

            tiles_output_folder = output_path / "tiles"
            tiles_output_folder.mkdir(parents=True, exist_ok=True)

            for i, tile in big_tiles_boundaries.iterrows():
                tile_output_file = tiles_output_folder / f"tile_{i}.nc"
                bounds = tile["geometry"].bounds
                minx, miny, maxx, maxy = bounds
                self.run_command(
                    f"gdalwarp -te {minx} {miny} {maxx} {maxy} -tr 250 250 -t_srs EPSG:3978 {str(final_output_file)} {str(tile_output_file)}"
                )
            # self.cleanup_tmp_folder(tmp_folder_path)
        except BaseException as e:
            logger.error(f"Error: {e}")
            self.cleanup_dataset_folder(dataset_folder_path)
            raise e

    def run_command(self, command):
        subprocess.run(command, shell=True, check=True)

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
        target_year_start_inclusive = periods_config["target_year_start_inclusive"]
        target_year_end_inclusive = periods_config["target_year_end_inclusive"]
        target_period_length_in_years = periods_config["target_period_length_in_years"]
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

    async def generate_targets(
        self,
        dataset_folder_path: Path,
        tmp_path: Path,
        target_years_ranges: list,
        big_tiles_boundaries: gpd.GeoDataFrame,
        resolution_config: dict,
        projections_config: dict,
    ):
        logger.info("Generating targets...")

        fire_data_source = NbacFireDataSource(Path(self.input_folder_path))

        tmp_target_folder_path = tmp_path / Path("targets")

        target = FireOccurrenceTarget(
            fire_data_source=fire_data_source,
            boundary=self.canada_boundary,
            target_pixel_size_in_meters=resolution_config["pixel_size_in_meters"],
            target_srid=projections_config["target_srid"],
            output_folder_path=tmp_target_folder_path,
            output_format=self.output_format,
            max_io_concurrency=self.max_io_concurrency,
            max_cpu_concurrency=self.max_cpu_concurrency,
        )

        target_ranges_combined_raster = await target.generate_target_for_years_ranges(
            target_years_ranges
        )

        logger.info("Generating tiles for targets...")

        dataset_targets_output_folder_path = dataset_folder_path / Path("target")
        dataset_targets_output_folder_path.mkdir(parents=True, exist_ok=True)

        for years_range, combined_raster_path in target_ranges_combined_raster.items():
            tiles_preprocessing_output_folder = (
                tmp_target_folder_path / f"{years_range[0]}_{years_range[-1]}/"
            )

            tiles_preprocessor = TilesPreprocessor(
                raw_tiles_folder=combined_raster_path.parent,
                tile_size_in_pixels=resolution_config["tile_size_in_pixels"],
                pixel_size_in_meters=resolution_config["pixel_size_in_meters"],
                output_folder=tiles_preprocessing_output_folder,
                big_tiles_boundaries=big_tiles_boundaries,
                input_format=self.output_format,
                output_format=self.output_format,
                layer_name="",
                source_srid=projections_config["target_srid"],
                target_srid=projections_config["target_srid"],
                resample_algorithm_continuous=resolution_config[
                    "resample_algorithm_continuous"
                ],
                resample_algorithm_categorical=resolution_config[
                    "resample_algorithm_categorical"
                ],
            )

            logger.info(f"Preprocessing tiles for target years range: {years_range}...")
            tiles_paths = await tiles_preprocessor.preprocess_tiles(
                data_type="categorical"
            )
            logger.info(
                f"Generated {len(tiles_paths)} tiles for target years range: {years_range}!"
            )

            tiles_folder_path = tiles_paths[0].parent

            years_range_output_folder_path = dataset_targets_output_folder_path / Path(
                f"{years_range[0]}_{years_range[-1]}"
            )
            years_range_output_folder_path.mkdir(parents=True, exist_ok=True)

            logger.info(
                f"Moving preprocessed tiles to final destination for target years range: {years_range}..."
            )
            shutil.move(tiles_folder_path, years_range_output_folder_path)

        logger.info("Generating targets DONE !")
        if self.debug:
            logs_folder = dataset_folder_path / Path("logs")
            logs_folder.mkdir(parents=True, exist_ok=True)
            logger.info("Saving target ranges combined raster logs for debugging...")
            serializable_target_ranges_combined_raster = {
                str(years_range): str(combined_raster_path)
                for years_range, combined_raster_path in target_ranges_combined_raster.items()
            }
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
