import numpy as np
import asyncio
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from loguru import logger
from boundaries.canada_boundary import CanadaBoundary
from data_sources.nbac_fire_data_source import NbacFireDataSource
from osgeo import gdal, osr
from raster_io.read import get_extension
from logging_utils.logging import get_ram_total, get_ram_used


class FireOccurrenceTarget:
    def __init__(
        self,
        fire_data_source: NbacFireDataSource,
        boundary: CanadaBoundary,
        target_pixel_size_in_meters: int,
        target_srid: int = 3978,
        output_folder_path: Path = Path("../data/target/"),
        output_format: str = "GTiff",
        max_io_concurrency: int = 16,
        max_cpu_concurrency: int = 8,
    ):
        self.fire_data_source = fire_data_source
        self.boundary = boundary
        self.target_pixel_size_in_meters = target_pixel_size_in_meters
        self.target_srid = target_srid
        self.output_folder_path = output_folder_path
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        self.output_format = output_format
        self.max_io_concurrency = max_io_concurrency
        self.max_cpu_concurrency = max_cpu_concurrency

        gdal.UseExceptions()

    async def generate_target_for_years_ranges(self, years_ranges: list) -> dict:
        years = set()
        for years_range in years_ranges:
            for year in years_range:
                years.add(year)

        logger.info(f"Downloading fire polygons for all {len(years)} years...")
        logger.opt(lazy=True).debug(
            "RAM Usage: {used:.2f}/{total:.2f}",
            used=get_ram_used(),
            total=get_ram_total(),
        )

        io_semaphore = asyncio.Semaphore(self.max_io_concurrency)

        tasks = set()
        for year in years:
            async with io_semaphore:
                task = asyncio.create_task(self.download_year_fire_polygons(year))
                tasks.add(task)
                task.add_done_callback(tasks.discard)

        years_fire_polygons_paths = await asyncio.gather(*tasks, return_exceptions=True)

        years_fire_polygons_paths = {
            year: year_fire_polygons_path
            for year, year_fire_polygons_path in years_fire_polygons_paths
        }

        logger.info("Computing output bounds based on boundary...")
        logger.opt(lazy=True).debug(
            "RAM Usage: {used:.2f}/{total:.2f}",
            used=get_ram_used(),
            total=get_ram_total(),
        )

        x_min, y_min, x_max, y_max = self.boundary.boundary.total_bounds

        output_raster_width_in_pixels = int(
            (x_max - x_min) / self.target_pixel_size_in_meters
        )
        output_raster_height_in_pixels = int(
            (y_max - y_min) / self.target_pixel_size_in_meters
        )

        logger.info(
            f"Final raster will have dimensions ({output_raster_height_in_pixels} x {output_raster_width_in_pixels}) pixels"
        )

        args = [
            (
                year,
                x_min,
                y_max,
                output_raster_width_in_pixels,
                output_raster_height_in_pixels,
                years_fire_polygons_paths,
            )
            for year in years
        ]
        logger.info(f"Rasterizing fire polgyons for all {len(years)} years...")
        logger.opt(lazy=True).debug(
            "RAM Usage: {used:.2f}/{total:.2f}",
            used=get_ram_used(),
            total=get_ram_total(),
        )

        loop = asyncio.get_event_loop()
        with ProcessPoolExecutor(max_workers=self.max_cpu_concurrency) as executor:
            rasterize_tasks = [
                loop.run_in_executor(executor, self.rasterize_fire_polygons, *arg)
                for arg in args
            ]
            rasterized_fire_polygons_paths = await asyncio.gather(
                *rasterize_tasks, return_exceptions=True
            )

        rasterized_fire_polygons_paths = {
            year: rasterized_fire_polygons_path
            for year, rasterized_fire_polygons_path in rasterized_fire_polygons_paths
        }

        args = [
            (
                years_range,
                x_min,
                y_max,
                output_raster_width_in_pixels,
                output_raster_height_in_pixels,
                rasterized_fire_polygons_paths,
            )
            for years_range in years_ranges
        ]
        logger.info(f"Combining rasters for all {len(years_ranges)} years ranges...")
        logger.opt(lazy=True).debug(
            "RAM Usage: {used:.2f}/{total:.2f}",
            used=get_ram_used(),
            total=get_ram_total(),
        )

        with ProcessPoolExecutor(max_workers=self.max_cpu_concurrency) as executor:
            combine_tasks = [
                loop.run_in_executor(executor, self.combine_rasters, *arg)
                for arg in args
            ]
            years_ranges_combined_rasters = await asyncio.gather(
                *combine_tasks, return_exceptions=True
            )

        years_ranges_combined_rasters = {
            years_range: combined_raster_path
            for years_range, combined_raster_path in years_ranges_combined_rasters
        }

        logger.info("Target generation done!")
        logger.opt(lazy=True).debug(
            "RAM Usage: {used:.2f}/{total:.2f}",
            used=get_ram_used(),
            total=get_ram_total(),
        )

        return years_ranges_combined_rasters

    async def download_year_fire_polygons(self, year: int) -> tuple:

        result = await asyncio.to_thread(self.fire_data_source.download, year)
        year_fire_polygons = result.to_crs(epsg=self.target_srid)

        output_folder = self.output_folder_path / f"fire_polygons_{year}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{year}.shp"

        year_fire_polygons.to_file(str(output_path.resolve()))

        return year, output_path

    def rasterize_fire_polygons(
        self,
        year: int,
        x_min: float,
        y_max: float,
        output_raster_width_in_pixels: int,
        output_raster_height_in_pixels: int,
        years_fire_polygons_paths: dict,
    ) -> tuple:
        output_extension = get_extension(self.output_format)
        output_folder = self.output_folder_path / f"rasterized_fire_polygons_{year}"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_raster_path = output_folder / f"{year}{output_extension}"
        nb_bands = 1
        output_raster_ds = gdal.GetDriverByName(self.output_format).Create(
            str(output_raster_path.resolve()),
            output_raster_width_in_pixels,
            output_raster_height_in_pixels,
            nb_bands,
            gdal.GDT_Byte,
        )
        output_raster_ds.SetGeoTransform(
            (
                x_min,
                self.target_pixel_size_in_meters,
                0,
                y_max,
                0,
                -self.target_pixel_size_in_meters,
            )
        )

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.target_srid)
        output_raster_ds.SetProjection(srs.ExportToWkt())

        output_band = output_raster_ds.GetRasterBand(1)
        output_band.SetNoDataValue(0)

        year_fire_polygons_path = years_fire_polygons_paths[year]
        shp_ds = gdal.OpenEx(str(year_fire_polygons_path.resolve()), gdal.OF_VECTOR)
        shp_layer = shp_ds.GetLayer()

        gdal.RasterizeLayer(
            output_raster_ds,
            [1],
            shp_layer,
            burn_values=[1],
        )

        del shp_ds
        del output_raster_ds

        return year, output_raster_path

    def combine_rasters(
        self,
        years_range: range,
        x_min: float,
        y_max: float,
        output_raster_width_in_pixels: int,
        output_raster_height_in_pixels: int,
        rasterized_fire_polygons_paths: dict,
    ) -> tuple:
        raster_paths = [rasterized_fire_polygons_paths[year] for year in years_range]

        combined_raster_data = np.zeros(
            (output_raster_height_in_pixels, output_raster_width_in_pixels),
            dtype=np.uint8,
        )

        for raster_path in raster_paths:
            raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
            raster_band = raster_ds.GetRasterBand(1)
            raster_data = raster_band.ReadAsArray()
            combined_raster_data = np.maximum(combined_raster_data, raster_data)
            del raster_ds

        driver = gdal.GetDriverByName(self.output_format)
        output_extension = get_extension(self.output_format)
        output_combined_raster_path = (
            self.output_folder_path
            / f"target_combined_{years_range[0]}_{years_range[-1]}"
            / f"combined{output_extension}"
        )
        output_combined_raster_path.parent.mkdir(parents=True, exist_ok=True)
        nb_bands = 1
        output_combined_raster_ds = driver.Create(
            str(output_combined_raster_path.resolve()),
            output_raster_width_in_pixels,
            output_raster_height_in_pixels,
            nb_bands,
            gdal.GDT_Byte,
        )
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.target_srid)
        output_combined_raster_ds.SetProjection(srs.ExportToWkt())
        output_combined_raster_ds.SetGeoTransform(
            (
                x_min,
                self.target_pixel_size_in_meters,
                0,
                y_max,
                0,
                -self.target_pixel_size_in_meters,
            )
        )
        output_band = output_combined_raster_ds.GetRasterBand(1)
        output_band.Fill(0)
        output_band.WriteArray(combined_raster_data)

        del output_combined_raster_ds

        return (years_range[0], years_range[-1]), output_combined_raster_path
