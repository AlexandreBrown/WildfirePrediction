import geopandas as gpd
import asyncio
from osgeo import gdal
from typing import Optional
from pathlib import Path
from raster_io.read import get_extension
from raster_io.read import get_formatted_file_path


class TilesPreprocessor:
    def __init__(
        self,
        raw_tiles_folder: Path,
        tile_size_in_pixels: int,
        pixel_size_in_meters: int,
        output_folder: Path,
        big_tiles_boundaries: gpd.GeoDataFrame,
        input_format: str = "netCDF",
        output_format: str = "GTiff",
        layer_name: Optional[str] = "",
        source_srid: int = 4326,
        target_srid: int = 3978,
        resample_algorithm_continuous: str = "lanczos",
        resample_algorithm_categorical: str = "nearest",
    ):
        self.tile_size_in_pixels = tile_size_in_pixels
        self.pixel_size_in_meters = pixel_size_in_meters
        self.output_folder = output_folder
        self.input_format = input_format
        self.output_format = output_format
        self.layer_name = layer_name
        self.source_srid = source_srid
        self.target_srid = target_srid
        self.resample_algorithm_continuous = resample_algorithm_continuous
        self.resample_algorithm_categorical = resample_algorithm_categorical
        self.raw_tiles_folder = raw_tiles_folder
        self.big_tiles_boundaries = big_tiles_boundaries

        gdal.UseExceptions()

        self.output_folder.mkdir(parents=True, exist_ok=True)

    async def preprocess_tiles(self, data_type: str) -> list:

        merged_raw_tiles_ds = await asyncio.to_thread(self.merge_raw_tiles)

        reprojected_raster_output_file = await asyncio.to_thread(
            self.resize_pixels_and_reproject, merged_raw_tiles_ds, data_type
        )

        tiles_paths = await asyncio.to_thread(
            self.make_tiles, reprojected_raster_output_file
        )

        return tiles_paths

    def merge_raw_tiles(self) -> gdal.Dataset:

        raw_tiles_paths = self.get_raw_tiles_paths()

        raw_tile_ds = gdal.Open(raw_tiles_paths[0], gdal.GA_ReadOnly)
        raw_band = raw_tile_ds.GetRasterBand(1)

        no_data_value = self.get_no_data_value(raw_band)

        vrt_options = gdal.BuildVRTOptions(
            separate=False,
            strict=True,
            srcNodata=no_data_value,
            VRTNodata=no_data_value,
        )

        raw_tiles_merged_output_path = self.output_folder / "raw_tiles_merged.vrt"

        return gdal.BuildVRT(
            destName=str(raw_tiles_merged_output_path.resolve()),
            srcDSOrSrcDSTab=raw_tiles_paths,
            options=vrt_options,
        )

    def get_raw_tiles_paths(self) -> list:
        extension = get_extension(self.input_format)

        raw_tiles_paths = [
            get_formatted_file_path(file_path, self.layer_name)
            for file_path in self.raw_tiles_folder.glob(f"*{extension}")
        ]

        return raw_tiles_paths

    def get_no_data_value(self, band: gdal.Band):
        if band.GetNoDataValue() is None:
            return None

        if band.DataType == gdal.GDT_Float32 or band.DataType == gdal.GDT_Float64:
            no_data_value = float(band.GetNoDataValue())
        else:
            no_data_value = int(band.GetNoDataValue())

        return no_data_value

    def resize_pixels_and_reproject(
        self, input_ds: gdal.Dataset, data_type: str
    ) -> Path:

        resampe_algorithm = self.get_resample_algorithm(data_type)

        no_data_value = self.get_no_data_value(input_ds.GetRasterBand(1))

        warp_options = gdal.WarpOptions(
            srcSRS=f"EPSG:{self.source_srid}",
            dstSRS=f"EPSG:{self.target_srid}",
            xRes=self.pixel_size_in_meters,
            yRes=self.pixel_size_in_meters,
            resampleAlg=resampe_algorithm,
            format=self.output_format,
            srcNodata=no_data_value,
            dstNodata=no_data_value,
            multithread=False,
        )

        extension = get_extension(self.output_format)

        reprojected_output_path = self.output_folder / f"reprojected_merged{extension}"

        dest_path = str(reprojected_output_path.resolve())

        # with gdal.config_options(
        #     {"GDAL_NUM_THREADS": "ALL_CPUS"}
        # ):
        gdal.Warp(dest_path, input_ds, options=warp_options)

        return reprojected_output_path

    def get_resample_algorithm(self, data_type: str) -> str:
        if data_type == "continuous":
            resample_algorithm = self.resample_algorithm_continuous
        elif data_type == "categorical":
            resample_algorithm = self.resample_algorithm_categorical
        else:
            raise ValueError(f"Unknown data type: {data_type}")

        return resample_algorithm

    def make_tiles(self, input_dataset_file_path: Path) -> list:
        tile_folder = self.output_folder / "tiles/"
        tile_folder.mkdir(parents=True, exist_ok=True)

        args = [
            (input_dataset_file_path, row, tile_folder, str(index))
            for index, row in self.big_tiles_boundaries.iterrows()
        ]

        return [self.make_tile(*arg) for arg in args]

    def make_tile(
        self,
        input_dataset_file_path: Path,
        row: gpd.GeoSeries,
        tile_folder: Path,
        tile_identifier: str,
    ) -> Path:
        tile_geometry = row["geometry"]
        bounds = tile_geometry.bounds
        minx, miny, maxx, maxy = bounds

        translate_options = gdal.TranslateOptions(
            format=self.output_format,
            outputType=gdal.GDT_Float32,
            width=self.tile_size_in_pixels,
            height=self.tile_size_in_pixels,
            projWin=[minx, maxy, maxx, miny],
            projWinSRS=f"EPSG:{self.target_srid}",
            strict=True,
            unscale=True,
        )

        extension = get_extension(self.output_format)

        tile_nc_file = tile_folder / f"tile_{tile_identifier}{extension}"

        input_ds = gdal.Open(str(input_dataset_file_path.resolve()), gdal.GA_ReadOnly)

        result = gdal.Translate(str(tile_nc_file), input_ds, options=translate_options)

        assert result is not None, f"Failed to create tile {tile_nc_file}"

        return tile_nc_file
