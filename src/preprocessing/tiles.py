import geopandas as gpd
import numpy as np
import logging
from osgeo import gdal
from osgeo import osr
from pathlib import Path
from typing import Optional


logging.basicConfig(level=logging.INFO)


class Tiles:
    def __init__(
        self,
        raw_tiles_folder: Path,
        layer_name: str,
        tile_size_in_pixels: int,
        pixel_size_in_meters: int,
        output_folder: Path,
        big_tiles: gpd.GeoDataFrame,
        source_srs: int = 4326,
        target_srs: int = 3978,
        resample_algorithm_continuous: str = "bilinear",
        resample_algorithm_categorical: str = "nearest"
    ):
        self.layer_name = layer_name
        self.tile_size_in_pixels = tile_size_in_pixels
        self.pixel_size_in_meters = pixel_size_in_meters
        self.output_folder = output_folder
        self.source_srs = source_srs
        self.target_srs = target_srs
        self.resample_algorithm_continuous = resample_algorithm_continuous
        self.resample_algorithm_categorical = resample_algorithm_categorical
        self.raw_tiles_folder = raw_tiles_folder
        self.big_tiles = big_tiles
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def generate_preprocessed_tiles(self, data_type: str) -> list:
        logging.info("Generating preprocessed tiles...")
        
        merged_raw_tiles_ds = self.merge_raw_tiles()
        reprojected_ds = self.resize_pixels_and_reproject(merged_raw_tiles_ds, data_type)
        tiles_paths = self.make_tiles(reprojected_ds)
        
        logging.info(f"Created {len(tiles_paths)} tiles!")
        return tiles_paths
    
    def merge_raw_tiles(self) -> gdal.Dataset:
        logging.info("Merging raw tiles...")
        
        raw_tiles_merged_output_path = self.output_folder / "raw_tiles_merged.vrt"
        
        raw_netcdf_file_paths = list(self.raw_tiles_folder.glob("*.nc"))

        formatted_raw_tiles_netcdf_paths = [f"NETCDF:\"{netcdf_file_path.resolve()}\"{':' + self.layer_name if self.layer_name != '' else ''}" for netcdf_file_path in raw_netcdf_file_paths]
        
        raw_tile_ds = gdal.Open(formatted_raw_tiles_netcdf_paths[0], gdal.GA_ReadOnly)
        raw_band = raw_tile_ds.GetRasterBand(1)
        
        no_data_value = self.get_no_data_value(raw_band)
        
        vrt_options = gdal.BuildVRTOptions(separate=False, strict=True, srcNodata=no_data_value, VRTNodata=no_data_value)
        
        return gdal.BuildVRT(str(raw_tiles_merged_output_path.resolve()), formatted_raw_tiles_netcdf_paths, options=vrt_options)

    def get_no_data_value(self, band: gdal.Band):
        if band.DataType == gdal.GDT_Float32 or band.DataType == gdal.GDT_Float64:
            no_data_value = float(band.GetNoDataValue())
        else:
            no_data_value = int(band.GetNoDataValue())
        
        return no_data_value

    def resize_pixels_and_reproject(self, input_ds: gdal.Dataset, data_type: str) -> gdal.Dataset:      
        logging.info("Resizing pixels and reprojecting to target projection...")

        reprojected_output_path = self.output_folder / "reprojected_merged.vrt"
        
        if data_type == "continuous":
            resample_algorithm = self.resample_algorithm_continuous
        elif data_type == "categorical":
            resample_algorithm = self.resample_algorithm_categorical
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        no_data_value = self.get_no_data_value(input_ds.GetRasterBand(1))
        
        warp_options = gdal.WarpOptions(
            srcSRS=f"EPSG:{self.source_srs}",
            dstSRS=f"EPSG:{self.target_srs}",
            xRes=self.pixel_size_in_meters,
            yRes=self.pixel_size_in_meters,
            resampleAlg=resample_algorithm,
            format='VRT',
            multithread=True,
            warpOptions=['NUM_THREADS=ALL_CPUS'],
            srcNodata=no_data_value,
            dstNodata=no_data_value
        )
        
        warped_ds = gdal.Warp(str(reprojected_output_path.resolve()), input_ds, options=warp_options)

        return warped_ds

    def make_tiles(self, input_ds: gdal.Dataset) -> list:
        logging.info("Generating tiles...")

        tile_folder = self.output_folder / "tiles/"
        tile_folder.mkdir(parents=True, exist_ok=True)

        num_bands = input_ds.RasterCount

        output_files_paths = []
        for _, row in self.big_tiles.iterrows():
            tile_output_path = self.make_tile(input_ds, num_bands, row, tile_folder)
            output_files_paths.append(tile_output_path)
    
        return output_files_paths

    def make_tile(self, input_ds: gdal.Dataset, num_bands: int, row: gpd.GeoSeries, tile_folder: Path) -> Optional[Path]:
        tile_geometry = row['geometry']
        bounds = tile_geometry.bounds
        minx, miny, maxx, maxy = bounds
        
        translate_options = gdal.TranslateOptions(
            format="netCDF",
            outputType=gdal.GDT_Float32,
            bandList=list(range(1, num_bands+1)),
            width=self.tile_size_in_pixels,
            height=self.tile_size_in_pixels,
            xRes=self.pixel_size_in_meters,
            yRes=self.pixel_size_in_meters,
            projWin=[minx, maxy, maxx, miny],
            projWinSRS=self.target_srs,
            strict=False,
            unscale=True
        )
        
        tile_nc_file = tile_folder / f"tile_{minx}_{maxy}.nc"
        
        gdal.Translate(destName=str(tile_nc_file.resolve()), srcDS=input_ds, options=translate_options)
        
        return tile_nc_file
