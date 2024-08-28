import numpy as np
from osgeo import gdal
from pathlib import Path


class ImageStats:
    def __init__(self):
        gdal.UseExceptions()

    def compute_aggregated_files_stats(self, file_paths: list) -> dict:
        files_stats = self.compute_files_stats(file_paths)

        aggregated_stats = {
            key: np.nanmean(files_stats[key], axis=0).tolist()
            for key in files_stats.keys()
        }

        return aggregated_stats

    def compute_files_stats(self, file_paths: list) -> dict:
        files_stats = {
            "mean": [],
            "std": [],
        }

        for file_path in file_paths:
            bands_stats = self.compute_bands_stats(file_path)

            for key in files_stats.keys():
                files_stats[key].append(bands_stats[key])

        return files_stats

    def compute_bands_stats(self, file_path: Path) -> dict:
        raster = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        bands = raster.RasterCount

        bands_stats = {
            "mean": [],
            "std": [],
        }

        for band in range(1, bands + 1):
            band = raster.GetRasterBand(band)
            data = band.ReadAsArray()
            no_data_value = band.GetNoDataValue()
            valid_data = data[data != no_data_value]

            if len(valid_data) == 0:
                band_mean = np.nan
                band_std = np.nan
            else:
                band_mean = np.mean(valid_data)
                band_std = np.std(valid_data)

            bands_stats["mean"].append(band_mean)
            bands_stats["std"].append(band_std)

        del raster

        return bands_stats
