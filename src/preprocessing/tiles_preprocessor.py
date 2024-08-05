import os
from venv import logger
import geopandas as gpd
from osgeo import gdal
from pathlib import Path
from multiprocessing import Pool


class TilesPreprocessor:
    def __init__(
        self,
        raw_tiles_folder: Path,
        layer_name: str,
        tile_size_in_pixels: int,
        pixel_size_in_meters: int,
        output_folder: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        source_srs: int = 4326,
        target_srs: int = 3978,
        resample_algorithm_continuous: str = "lanczos",
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
        self.big_tiles_boundaries = big_tiles_boundaries
        
        gdal.UseExceptions()
        
        self.output_folder.mkdir(parents=True, exist_ok=True)
    
    def preprocess_tiles(self, data_type: str) -> list:
                
        merged_raw_tiles_ds = self.merge_raw_tiles()
        reprojected_raster_output_file = self.resize_pixels_and_reproject(merged_raw_tiles_ds, data_type)
        tiles_paths = self.make_tiles(reprojected_raster_output_file)
        
        return tiles_paths
    
    def merge_raw_tiles(self) -> gdal.Dataset:
        
        logger.info("Merging raw tiles...")
        
        raw_tiles_merged_output_path = self.output_folder / "raw_tiles_merged.vrt"
        
        raw_netcdf_file_paths = list(self.raw_tiles_folder.glob("*.nc"))

        formatted_raw_tiles_netcdf_paths = [f"NETCDF:\"{netcdf_file_path.resolve()}\"{':' + self.layer_name if self.layer_name != '' else ''}" for netcdf_file_path in raw_netcdf_file_paths]
        
        raw_tile_ds = gdal.Open(formatted_raw_tiles_netcdf_paths[0], gdal.GA_ReadOnly)
        raw_band = raw_tile_ds.GetRasterBand(1)
        
        no_data_value = self.get_no_data_value(raw_band)
        
        vrt_options = gdal.BuildVRTOptions(separate=False, strict=True, srcNodata=no_data_value, VRTNodata=no_data_value)
        
        return gdal.BuildVRT(str(raw_tiles_merged_output_path.resolve()), formatted_raw_tiles_netcdf_paths, options=vrt_options)

    def get_no_data_value(self, band: gdal.Band):
        if band.GetNoDataValue() is None:
            return None
        
        if band.DataType == gdal.GDT_Float32 or band.DataType == gdal.GDT_Float64:
            no_data_value = float(band.GetNoDataValue())
        else:
            no_data_value = int(band.GetNoDataValue())
        
        return no_data_value

    def resize_pixels_and_reproject(self, input_ds: gdal.Dataset, data_type: str) -> Path:      

        logger.info(f"Resizing pixels and reprojecting to srs {self.target_srs}...")

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
            srcNodata=no_data_value,
            dstNodata=no_data_value
        )
        
        gdal.Warp(str(reprojected_output_path.resolve()), input_ds, options=warp_options)

        return reprojected_output_path

    def make_tiles(self, input_dataset_file_path: Path) -> list:
        tile_folder = self.output_folder / "tiles/"
        tile_folder.mkdir(parents=True, exist_ok=True)

        output_files_paths = []
        
        args = [(input_dataset_file_path, row, tile_folder, str(index)) for index, row in self.big_tiles_boundaries.iterrows()]
        
        max_nb_processes = min(max(1, (len(os.sched_getaffinity(0)) - 1)), len(args))
        
        logger.info(f"Making tiles using {max_nb_processes} processes...")
        
        with Pool(processes=max_nb_processes) as pool:
            output_files_paths = pool.starmap(self.make_tile, args)
    
        return output_files_paths

    def make_tile(self, input_dataset_file_path: Path, row: gpd.GeoSeries, tile_folder: Path, tile_identifier: str) -> Path:
        tile_geometry = row['geometry']
        bounds = tile_geometry.bounds
        minx, miny, maxx, maxy = bounds
        
        translate_options = gdal.TranslateOptions(
            format="netCDF",
            outputType=gdal.GDT_Float32,
            width=self.tile_size_in_pixels,
            height=self.tile_size_in_pixels,
            projWin=[minx, maxy, maxx, miny],
            projWinSRS=f"EPSG:{self.target_srs}",
            strict=True,
            unscale=True
        )
        
        tile_nc_file = tile_folder / f"tile_{tile_identifier}.nc"
        
        input_ds = gdal.Open(str(input_dataset_file_path.resolve()), gdal.GA_ReadOnly)
        
        result = gdal.Translate(destName=str(tile_nc_file), srcDS=input_ds, options=translate_options)
        
        assert result is not None, f"Failed to create tile {tile_nc_file}"
        
        return tile_nc_file
