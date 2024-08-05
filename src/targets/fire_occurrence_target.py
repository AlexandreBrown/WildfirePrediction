import logging
import numpy as np
import multiprocessing as mp
import os
from pathlib import Path
from boundaries.canada_boundary import CanadaBoundary
from data_sources.nbac_fire_data_source import NbacFireDataSource
from osgeo import gdal, osr


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class FireOccurrenceTarget:
    def __init__(
        self, 
        fire_data_source: NbacFireDataSource,
        boundary: CanadaBoundary,
        target_pixel_size_in_meters: int,
        target_srid: int = 3978,
        output_folder_path: Path = Path("../data/target/"),
        max_nb_processes: int = max(1, (len(os.sched_getaffinity(0)) - 1) //2)
    ):
        self.fire_data_source = fire_data_source
        self.boundary = boundary
        self.target_pixel_size_in_meters = target_pixel_size_in_meters
        self.target_srid = target_srid
        self.output_folder_path = output_folder_path
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        self.max_nb_processes = max_nb_processes
        gdal.UseExceptions()
    
    def generate_target_for_years_ranges(self, years_ranges: list) -> dict:
        
        years = set()
        for years_range in years_ranges:
            for year in years_range:
                years.add(year)
        
        nb_processes = min(self.max_nb_processes, len(years))
        logger.info(f"Downloading fire polygons for all years using {nb_processes} processes...")
        with mp.Pool(processes=nb_processes) as pool:
            years_fire_polygons_paths = pool.map(self.download_year_fire_polygons, years)
        
        years_fire_polygons_paths = {
            year: year_fire_polygons_path for year, year_fire_polygons_path in years_fire_polygons_paths
        }
        
        logger.info("Computing output bounds based on boundary...")
        x_min, y_min, x_max, y_max = self.boundary.boundary.total_bounds
        
        output_raster_width_in_pixels = int((x_max - x_min) / self.target_pixel_size_in_meters)
        output_raster_height_in_pixels = int((y_max - y_min) / self.target_pixel_size_in_meters)
        
        logger.info(f"Final raster will have dimensions ({output_raster_height_in_pixels} x {output_raster_width_in_pixels}) pixels")
        
        args = [(year, x_min, y_max, output_raster_width_in_pixels, output_raster_height_in_pixels, years_fire_polygons_paths) for year in years]
        nb_processes = min(self.max_nb_processes, len(args))
        logger.info(f"Rasterizing fire polgyons for all years using {nb_processes} processes...")
        with mp.Pool(processes=nb_processes) as pool:
            rasterized_fire_polygons_paths = pool.starmap(self.rasterize_fire_polygons, args)
        
        rasterized_fire_polygons_paths = {
            year: rasterized_fire_polygons_path for year, rasterized_fire_polygons_path in rasterized_fire_polygons_paths
        }
        
        args = [(years_range, x_min, y_max, output_raster_width_in_pixels, output_raster_height_in_pixels, rasterized_fire_polygons_paths) for years_range in years_ranges]
        nb_processes = min(self.max_nb_processes, len(args))
        logger.info(f"Combining rasters for all years ranges using {nb_processes} processes...")
        with mp.Pool(processes=nb_processes) as pool:
            years_ranges_combined_rasters = pool.starmap(self.combine_rasters, args)
        
        years_ranges_combined_rasters = {
            years_range: combined_raster_path for years_range, combined_raster_path in years_ranges_combined_rasters
        }
        
        return years_ranges_combined_rasters
    
    def download_year_fire_polygons(self, year: int) -> tuple:
        year_fire_polygons = self.fire_data_source.download(year).to_crs(epsg=self.target_srid)
        
        output_path = self.output_folder_path / f"{year}.shp"
        
        year_fire_polygons.to_file(str(output_path.resolve()))
        
        return year, output_path
    
    def rasterize_fire_polygons(
        self,
        year: int, 
        x_min: float,
        y_max: float,
        output_raster_width_in_pixels: int,
        output_raster_height_in_pixels: int,
        years_fire_polygons_paths: dict
    ) -> tuple:
        output_raster_path = self.output_folder_path / f"{year}.nc"
        nb_bands = 1
        output_raster_ds = gdal.GetDriverByName('netCDF').Create(str(output_raster_path.resolve()), output_raster_width_in_pixels, output_raster_height_in_pixels, nb_bands, gdal.GDT_Byte)
        output_raster_ds.SetGeoTransform((x_min, self.target_pixel_size_in_meters, 0, y_max, 0, -self.target_pixel_size_in_meters))

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
        
        return year, output_raster_path
    
    def combine_rasters(
        self, 
        years_range: range,
        x_min: float, 
        y_max: float,
        output_raster_width_in_pixels: int,
        output_raster_height_in_pixels: int,
        rasterized_fire_polygons_paths: dict
    ) -> tuple:
        raster_paths = [rasterized_fire_polygons_paths[year] for year in years_range]
        
        combined_raster_data = np.zeros((output_raster_height_in_pixels, output_raster_width_in_pixels), dtype=np.uint8)

        for raster_path in raster_paths:
            raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
            raster_band = raster_ds.GetRasterBand(1)
            raster_data = raster_band.ReadAsArray()
            combined_raster_data = np.maximum(combined_raster_data, raster_data)

        driver = gdal.GetDriverByName('netCDF')
        output_combined_raster_path = self.output_folder_path / f"target_{years_range[0]}_{years_range[-1]}" / "combined.nc"
        output_combined_raster_path.parent.mkdir(parents=True, exist_ok=True)
        nb_bands = 1
        output_combined_raster_ds = driver.Create(str(output_combined_raster_path.resolve()), output_raster_width_in_pixels, output_raster_height_in_pixels, nb_bands, gdal.GDT_Byte)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.target_srid)
        output_combined_raster_ds.SetProjection(srs.ExportToWkt())
        output_combined_raster_ds.SetGeoTransform((x_min, self.target_pixel_size_in_meters, 0, y_max, 0, -self.target_pixel_size_in_meters))
        output_band = output_combined_raster_ds.GetRasterBand(1)
        output_band.Fill(0)
        output_band.WriteArray(combined_raster_data)
        
        return (years_range[0], years_range[-1]), output_combined_raster_path
