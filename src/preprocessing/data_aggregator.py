import numpy as np
import asyncio
from osgeo import gdal
from pathlib import Path
from functools import partial
from raster_io.read import get_extension


def update_sum_count(sum_data, count_data, no_data_value, band_data):
    valid_mask = band_data != no_data_value
    sum_data[valid_mask] += band_data[valid_mask]
    count_data[valid_mask] += 1


def compute_avg_data(sum_data, count_data, no_data_value):
    avg_data = np.zeros_like(sum_data, dtype=np.float32)
    np.divide(sum_data, count_data, out=avg_data, where=count_data != 0)
    avg_data[count_data == 0] = no_data_value
    return avg_data


def update_max_data(max_data, no_data_value, band_data):
    previously_no_data_mask = max_data == no_data_value
    max_data[previously_no_data_mask] = band_data[previously_no_data_mask]
    valid_mask = band_data != no_data_value
    max_data[valid_mask] = np.maximum(max_data[valid_mask], band_data[valid_mask])


class DataAggregator:
    def __init__(self, output_format: str):
        self.output_format = output_format

        gdal.UseExceptions()
        cache_max_in_bytes = 128_000_000
        gdal.SetCacheMax(cache_max_in_bytes)

    async def aggregate_bands_by_average(
        self, input_dataset_path: Path, output_folder_path: Path
    ) -> Path:
        input_dataset = gdal.Open(str(input_dataset_path))

        output_dataset, output_band, output_file_path = await asyncio.to_thread(
            self.create_aggregated_dataset_and_band,
            input_dataset,
            output_folder_path,
            input_dataset_path.stem,
        )

        no_data_value = output_band.GetNoDataValue()
        sum_data = np.zeros(
            (output_dataset.RasterYSize, output_dataset.RasterXSize), dtype=np.float32
        )
        count_data = np.zeros(
            (output_dataset.RasterYSize, output_dataset.RasterXSize), dtype=np.int64
        )

        on_band_read = partial(
            update_sum_count,
            sum_data=sum_data,
            count_data=count_data,
            no_data_value=no_data_value,
        )
        get_final_output_band_data = partial(
            compute_avg_data,
            sum_data=sum_data,
            count_data=count_data,
            no_data_value=no_data_value,
        )
        await asyncio.to_thread(
            self.aggregate_bands,
            input_dataset,
            output_band,
            on_band_read,
            get_final_output_band_data,
            output_dataset,
        )

        del input_dataset
        del output_dataset

        return output_file_path

    async def aggregate_bands_by_max(
        self, input_dataset_path: Path, output_folder_path: Path
    ) -> Path:
        input_dataset = gdal.Open(str(input_dataset_path))

        output_dataset, output_band, output_file_path = await asyncio.to_thread(
            self.create_aggregated_dataset_and_band,
            input_dataset,
            output_folder_path,
            input_dataset_path.stem,
        )

        no_data_value = output_band.GetNoDataValue()
        max_data = np.zeros(
            (output_dataset.RasterYSize, output_dataset.RasterXSize), dtype=np.float32
        )

        on_band_read = partial(
            update_max_data, max_data=max_data, no_data_value=no_data_value
        )
        get_final_output_band_data = lambda: max_data

        await asyncio.to_thread(
            self.aggregate_bands,
            input_dataset,
            output_band,
            on_band_read,
            get_final_output_band_data,
            output_dataset,
        )

        del input_dataset
        del output_dataset

        return output_file_path

    def aggregate_bands(
        self,
        input_dataset: gdal.Dataset,
        output_band: gdal.Band,
        on_band_read,
        get_final_output_band_data,
        output_dataset: gdal.Dataset,
    ):
        has_sub_datasets = len(input_dataset.GetSubDatasets()) > 0
        num_bands = self.get_num_bands(input_dataset, has_sub_datasets)

        for band_index in range(1, num_bands + 1):
            band_data = self.get_band_data(input_dataset, has_sub_datasets, band_index)
            on_band_read(band_data=band_data)

        output_band.WriteArray(get_final_output_band_data())

        output_dataset.FlushCache()

    def create_aggregated_dataset_and_band(
        self,
        input_dataset: gdal.Dataset,
        output_path: Path,
        output_file_name_without_extension: str,
    ) -> tuple:

        has_sub_datasets = len(input_dataset.GetSubDatasets()) > 0

        no_data_value = self.get_no_data_value(input_dataset, has_sub_datasets)
        xsize = self.get_xsize(input_dataset, has_sub_datasets)
        ysize = self.get_ysize(input_dataset, has_sub_datasets)

        driver = gdal.GetDriverByName(self.output_format)
        output_path.mkdir(parents=True, exist_ok=True)
        output_extension = get_extension(self.output_format)
        output_file_path = (
            output_path / f"{output_file_name_without_extension}{output_extension}"
        )
        output_dataset = driver.Create(
            str(output_file_path.resolve()),
            xsize=xsize,
            ysize=ysize,
            bands=1,
            eType=gdal.GDT_Float32,
        )
        output_dataset.SetGeoTransform(
            self.get_geotransform(input_dataset, has_sub_datasets)
        )
        output_dataset.SetProjection(
            self.get_projection(input_dataset, has_sub_datasets)
        )
        output_band = output_dataset.GetRasterBand(1)
        output_band.SetNoDataValue(no_data_value)

        return output_dataset, output_band, output_file_path

    def get_num_bands(self, dataset: gdal.Dataset, has_sub_datasets: bool) -> int:
        num_bands = None

        if has_sub_datasets:
            num_bands = len(dataset.GetSubDatasets())
        else:
            num_bands = dataset.RasterCount

        return num_bands

    def get_no_data_value(self, dataset: gdal.Dataset, has_sub_datasets: bool) -> float:
        no_data_value = None

        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            no_data_value = first_band_dataset.GetRasterBand(1).GetNoDataValue()
            del first_band_dataset
        else:
            no_data_value = dataset.GetRasterBand(1).GetNoDataValue()

        return no_data_value

    def get_xsize(self, dataset: gdal.Dataset, has_sub_datasets: bool) -> int:
        xsize = None

        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            xsize = first_band_dataset.RasterXSize
            del first_band_dataset
        else:
            xsize = dataset.RasterXSize

        return xsize

    def get_ysize(self, dataset: gdal.Dataset, has_sub_datasets: bool) -> int:
        ysize = None

        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            ysize = first_band_dataset.RasterYSize
            del first_band_dataset
        else:
            ysize = dataset.RasterYSize

        return ysize

    def get_geotransform(self, dataset: gdal.Dataset, has_sub_datasets: bool):
        geotransform = None

        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            geotransform = first_band_dataset.GetGeoTransform()
            del first_band_dataset
        else:
            geotransform = dataset.GetGeoTransform()

        return geotransform

    def get_projection(self, dataset: gdal.Dataset, has_sub_datasets: bool):
        projection = None

        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            projection = first_band_dataset.GetProjection()
            del first_band_dataset
        else:
            projection = dataset.GetProjection()

        return projection

    def get_band_data(
        self, dataset: gdal.Dataset, has_sub_datasets: bool, band_index: int
    ) -> np.ndarray:
        band_data = None

        if has_sub_datasets:
            band_ds = gdal.Open(dataset.GetSubDatasets()[band_index - 1][0])
            band_data = band_ds.ReadAsArray()
            del band_ds
        else:
            band_data = dataset.GetRasterBand(1).ReadAsArray()

        return band_data

    async def aggregate_files_by_average(
        self, input_datasets_paths: list, output_folder_path: Path
    ) -> Path:
        input_datasets = [
            gdal.Open(str(input_path)) for input_path in input_datasets_paths
        ]
        output_dataset, output_band, output_file_path = await asyncio.to_thread(
            self.create_aggregated_dataset_and_band,
            input_datasets[0],
            output_folder_path,
            input_datasets_paths[0].stem,
        )

        no_data_value = output_band.GetNoDataValue()
        sum_data = np.zeros(
            (output_dataset.RasterYSize, output_dataset.RasterXSize), dtype=np.float32
        )
        count_data = np.zeros(
            (output_dataset.RasterYSize, output_dataset.RasterXSize), dtype=np.int32
        )

        on_band_read = partial(
            update_sum_count,
            sum_data=sum_data,
            count_data=count_data,
            no_data_value=no_data_value,
        )
        get_final_output_band_data = partial(
            compute_avg_data,
            sum_data=sum_data,
            count_data=count_data,
            no_data_value=no_data_value,
        )
        await asyncio.to_thread(
            self.aggregate_datasets,
            input_datasets,
            output_band,
            on_band_read,
            get_final_output_band_data,
            output_dataset,
        )

        for dataset in input_datasets:
            del dataset
        del output_dataset

        return output_file_path

    def aggregate_datasets(
        self,
        input_datasets: list,
        output_band: gdal.Band,
        on_band_read,
        get_final_output_band_data,
        output_dataset: gdal.Dataset,
    ):

        for input_dataset in input_datasets:
            band_data = input_dataset.GetRasterBand(1).ReadAsArray()
            on_band_read(band_data=band_data)

        output_band.WriteArray(get_final_output_band_data())
        output_dataset.FlushCache()

    async def aggregate_files_by_max(
        self, input_datasets_paths: list, output_folder_path: Path
    ) -> Path:
        input_datasets = [
            gdal.Open(str(input_path)) for input_path in input_datasets_paths
        ]
        output_dataset, output_band, output_file_path = await asyncio.to_thread(
            self.create_aggregated_dataset_and_band,
            input_datasets[0],
            output_folder_path,
            input_datasets_paths[0].stem,
        )

        no_data_value = output_band.GetNoDataValue()
        max_data = np.zeros(
            (output_dataset.RasterYSize, output_dataset.RasterXSize), dtype=np.float32
        )

        on_band_read = partial(
            update_max_data, max_data=max_data, no_data_value=no_data_value
        )
        get_final_output_band_data = lambda: max_data
        await asyncio.to_thread(
            self.aggregate_datasets,
            input_datasets,
            output_band,
            on_band_read,
            get_final_output_band_data,
            output_dataset,
        )

        for dataset in input_datasets:
            del dataset
        del output_dataset

        return output_file_path
