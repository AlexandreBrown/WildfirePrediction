import numpy as np
import logging
from osgeo import gdal
from osgeo import osr
from pathlib import Path


logging.basicConfig(level=logging.INFO)


class Tiles:
    def __init__(
        self,
        raw_tiles_folder: Path,
        layer_name: str,
        tile_size_in_pixels: int,
        pixel_size_in_meters: int,
        output_folder: Path,
        source_srs: int = 4326,
        target_srs: int = 3978,
        resample_algorithm: str = "bilinear",
    ):
        self.layer_name = layer_name
        self.tile_size_in_pixels = tile_size_in_pixels
        self.pixel_size_in_meters = pixel_size_in_meters
        self.output_folder = output_folder
        self.source_srs = source_srs
        self.target_srs = target_srs
        self.resample_algorithm = resample_algorithm
        self.raw_tiles_folder = raw_tiles_folder
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def generate_preprocessed_tiles(self) -> list:
        merged_raw_tiles_ds = self.merge_raw_tiles()
        reprojected_ds = self.resize_pixels_and_reproject(merged_raw_tiles_ds)
        tiles_paths = self.make_tiles(reprojected_ds)
        logging.info(f"Created {len(tiles_paths)} tiles!")
        return tiles_paths
    
    def merge_raw_tiles(self) -> gdal.Dataset:
        logging.info("Merging raw tiles...")
        
        raw_tiles_merged_output_path = self.output_folder / "raw_tiles_merged.vrt"
        
        raw_netcdf_file_paths = list(self.raw_tiles_folder.glob("*.nc"))

        formatted_raw_tiles_netcdf_paths = [f"NETCDF:\"{netcdf_file_path.resolve()}\":{self.layer_name}" for netcdf_file_path in raw_netcdf_file_paths]
        
        vrt_options = gdal.BuildVRTOptions(separate=False, strict=True)
        
        return gdal.BuildVRT(str(raw_tiles_merged_output_path.resolve()), formatted_raw_tiles_netcdf_paths, options=vrt_options)

    def resize_pixels_and_reproject(self, input_ds: gdal.Dataset) -> gdal.Dataset:      
        logging.info("Resizing pixels and reprojecting to target projection...")

        reprojected_output_path = self.output_folder / "reprojected_merged.vrt"
        warp_options = gdal.WarpOptions(
            srcSRS=f"EPSG:{self.source_srs}",
            dstSRS=f"EPSG:{self.target_srs}",
            xRes=self.pixel_size_in_meters,
            yRes=self.pixel_size_in_meters,
            resampleAlg=self.resample_algorithm,
            format='VRT',
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS']
        )
        
        return gdal.Warp(str(reprojected_output_path.resolve()), input_ds, options=warp_options)

    def make_tiles(self, input_ds: gdal.Dataset) -> list:
        logging.info("Generating tiles...")

        tile_folder = self.output_folder / "tiles/"
        tile_folder.mkdir(parents=True, exist_ok=True)

        num_bands = input_ds.RasterCount

        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(self.target_srs)

        geotransform = input_ds.GetGeoTransform()

        output_files_paths = []
        for x_offset in range(0, input_ds.RasterXSize, self.tile_size_in_pixels):
            for y_offset in range(0, input_ds.RasterYSize, self.tile_size_in_pixels):
                logging.info(f"Generating tile for x_offset: {x_offset}, y_offset: {y_offset}")
                
                tile_nc_file = tile_folder / f"tile_{x_offset}_{y_offset}.nc"
                
                driver = gdal.GetDriverByName("netCDF")
                tile_ds = driver.Create(str(tile_nc_file), self.tile_size_in_pixels, self.tile_size_in_pixels, num_bands, gdal.GDT_Float32)

                new_geotransform = (
                    geotransform[0] + x_offset * geotransform[1],
                    geotransform[1],
                    0,
                    geotransform[3] + y_offset * geotransform[5],
                    0,
                    geotransform[5]
                )
                tile_ds.SetGeoTransform(new_geotransform)
                tile_ds.SetProjection(target_srs.ExportToWkt())

                keep_tile = False
                for band in range(1, num_bands+1):
                    raster_band = input_ds.GetRasterBand(band)
                    read_width = min(self.tile_size_in_pixels, input_ds.RasterXSize - x_offset)
                    read_height = min(self.tile_size_in_pixels, input_ds.RasterYSize - y_offset)
                    actual_data = raster_band.ReadAsArray(x_offset, y_offset, read_width, read_height)
                    no_data_value = raster_band.GetNoDataValue()
                    non_zero_data_count = np.count_nonzero(actual_data != no_data_value)
                    
                    keep_tile = non_zero_data_count > 0
                    if not keep_tile:
                        tile_nc_file.unlink()
                        break
                    
                    tile_band = tile_ds.GetRasterBand(band)
                    tile_band.SetNoDataValue(no_data_value)

                    tile_data = np.full((self.tile_size_in_pixels, self.tile_size_in_pixels), no_data_value, dtype=np.float32)
                    tile_data[:read_height, :read_width] = actual_data
                    
                    tile_band.WriteArray(tile_data)
                    tile_band.FlushCache()
                
                if keep_tile:
                    output_files_paths.append(tile_nc_file)
        
        return output_files_paths
