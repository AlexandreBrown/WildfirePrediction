import numpy as np
from osgeo import gdal
from pathlib import Path


class Aggregator:
    def aggregate_bands_by_average(self, input_path: Path, output_path: Path) -> Path:
        dataset = gdal.Open(str(input_path))
        
        has_sub_datasets = len(dataset.GetSubDatasets()) > 0
        
        num_bands = self.get_num_bands(dataset, has_sub_datasets)
        
        no_data_value = self.get_no_data_value(dataset, has_sub_datasets)
        xsize = self.get_xsize(dataset, has_sub_datasets)
        ysize = self.get_ysize(dataset, has_sub_datasets)

        driver = gdal.GetDriverByName("netCDF")
        output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_path / f"{input_path.stem}_aggregated.nc"
        output_dataset = driver.Create(
            str(output_file_path.resolve()), 
            xsize=xsize, 
            ysize=ysize, 
            bands=1, 
            eType=gdal.GDT_Float32
        )
        output_dataset.SetGeoTransform(self.get_geotransform(dataset, has_sub_datasets))
        output_dataset.SetProjection(self.get_projection(dataset, has_sub_datasets))
        output_band = output_dataset.GetRasterBand(1)
        output_band.SetNoDataValue(no_data_value)
        
        sum_data = np.zeros((ysize, xsize), dtype=np.float32)
        count_data = np.zeros((ysize, xsize), dtype=np.int64)

        for band_index in range(1, num_bands+1):
            band_data =  self.get_band_data(dataset, has_sub_datasets, band_index)
            
            valid_mask = band_data != no_data_value
            
            sum_data[valid_mask] += band_data[valid_mask]
            count_data[valid_mask] += 1

        avg_data = np.zeros_like(sum_data, dtype=np.float32)
        np.divide(sum_data, count_data, out=avg_data, where=count_data != 0)
        avg_data[count_data == 0] = no_data_value

        output_band.WriteArray(avg_data)
        output_band.FlushCache()
        
        return output_file_path
    
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
        else:
            no_data_value = dataset.GetRasterBand(1).GetNoDataValue()
        
        return no_data_value
    
    def get_xsize(self, dataset: gdal.Dataset, has_sub_datasets: bool) -> int:
        xsize = None
        
        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            xsize = first_band_dataset.RasterXSize
        else:
            xsize = dataset.RasterXSize
        
        return xsize
    
    def get_ysize(self, dataset: gdal.Dataset, has_sub_datasets: bool) -> int:
        ysize = None
        
        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            ysize = first_band_dataset.RasterYSize
        else:
            ysize = dataset.RasterYSize
        
        return ysize
    
    def get_geotransform(self, dataset: gdal.Dataset, has_sub_datasets: bool):
        geotransform = None
        
        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            geotransform = first_band_dataset.GetGeoTransform()
        else:
            geotransform = dataset.GetGeoTransform()
        
        return geotransform

    def get_projection(self, dataset: gdal.Dataset, has_sub_datasets: bool):
        projection = None
        
        if has_sub_datasets:
            first_band_dataset = gdal.Open(dataset.GetSubDatasets()[0][0])
            projection = first_band_dataset.GetProjection()
        else:
            projection = dataset.GetProjection()
        
        return projection
    
    def get_band_data(self, dataset: gdal.Dataset, has_sub_datasets: bool, band_index: int) -> np.ndarray:
        band_data = None

        if has_sub_datasets:
            band_data = gdal.Open(dataset.GetSubDatasets()[band_index -1][0]).ReadAsArray()
        else:
            band_data = dataset.GetRasterBand(1).ReadAsArray()
        
        return band_data
    
    def aggregate_files_by_average(self, input_paths: list, output_path: Path, output_file_name_without_extension: str) -> Path:
        datasets = [gdal.Open(str(input_path)) for input_path in input_paths]
        
        no_data_value = self.get_no_data_value(datasets[0], has_sub_datasets=False)
        xsize = self.get_xsize(datasets[0], has_sub_datasets=False)
        ysize = self.get_ysize(datasets[0], has_sub_datasets=False)

        driver = gdal.GetDriverByName("netCDF")
        output_path.mkdir(parents=True, exist_ok=True)
        output_file_path = output_path / f"{input_paths[0].stem}_aggregated.nc"
        output_dataset = driver.Create(
            str(output_file_path.resolve()), 
            xsize=xsize, 
            ysize=ysize, 
            bands=1, 
            eType=gdal.GDT_Float32
        )
        output_dataset.SetGeoTransform(datasets[0].GetGeoTransform())
        output_dataset.SetProjection(datasets[0].GetProjection())
        output_band = output_dataset.GetRasterBand(1)
        output_band.SetNoDataValue(no_data_value)
        
        sum_data = np.zeros((ysize, xsize), dtype=np.float32)
        count_data = np.zeros((ysize, xsize), dtype=np.int64)

        for dataset in datasets:
            band_data = dataset.GetRasterBand(1).ReadAsArray()
            
            valid_mask = band_data != no_data_value
            
            sum_data[valid_mask] += band_data[valid_mask]
            count_data[valid_mask] += 1

        avg_data = np.zeros_like(sum_data, dtype=np.float32)
        np.divide(sum_data, count_data, out=avg_data, where=count_data != 0)
        avg_data[count_data == 0] = no_data_value

        output_band.WriteArray(avg_data)
        output_band.FlushCache()
        
        return output_file_path
