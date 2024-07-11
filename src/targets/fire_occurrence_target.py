import geopandas as gpd
import shutil
import numpy as np
from pathlib import Path
from data_sources.nbac_fire_data_source import NbacFireDataSource
from boundaries.canada_boundary import CanadaBoundary
from osgeo import gdal, osr


class FireOccurrenceTarget:
    def __init__(
        self, 
        fire_data_source: NbacFireDataSource, 
        canada: CanadaBoundary,
        resolution_in_meters: int,
        target_epsg_code: int = 3978,
        raw_data_output_folder_path: Path = Path("../data/raw/"),
        target_data_output_folder_path: Path = Path("../data/target/")
    ):
        self.fire_data_source = fire_data_source
        self.canada = canada
        self.resolution_in_meters = resolution_in_meters
        self.target_epsg_code = target_epsg_code
        self.raw_data_output_folder_path = raw_data_output_folder_path
        self.raw_data_output_folder_path.mkdir(parents=True, exist_ok=True)
        self.target_data_output_folder_path = target_data_output_folder_path
        self.target_data_output_folder_path.mkdir(parents=True, exist_ok=True)
    
    def generate_targets(self, years: range) -> list:
        merged_raster = None
        
        for year in years:
            target_file_path = self.generate(year=year)
            target_dataset = gdal.Open(str(target_file_path))
            target_band = target_dataset.GetRasterBand(1)
            target_data = target_band.ReadAsArray()

            if merged_raster is None:
                rows, cols = target_data.shape
                merged_raster = np.zeros((rows, cols), dtype=np.uint8)
                target_srs = osr.SpatialReference()
                target_srs.ImportFromEPSG(self.target_epsg_code)
                geotransform = target_dataset.GetGeoTransform()

            merged_raster = np.maximum(merged_raster, target_data, dtype=np.uint8)
            
            target_file_path.unlink()

        merged_dataset = self.create_raster_from_array(merged_raster, geotransform, target_srs)

        merged_output_paths = []
        
        for year in years:
            merged_output_path = self.save_target(merged_dataset, year, self.target_data_output_folder_path)
            merged_output_paths.append(merged_output_path)
            
        return merged_output_paths
    
    def create_raster_from_array(self, array: np.ndarray, geotransform: tuple, srs: osr.SpatialReference) -> gdal.Dataset:
        driver = gdal.GetDriverByName('MEM')
        
        rows, cols = array.shape
        
        dataset = driver.Create('', cols, rows, 1, gdal.GDT_Byte)
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(srs.ExportToWkt())
        
        band = dataset.GetRasterBand(1)
        band.WriteArray(array)
        band.SetNoDataValue(0)
        
        return dataset
    
    def generate(self, year: int) -> Path:
        fire_polygons_shp_path = self.get_fire_polygons_path(year)
        
        canada_boundary = self.get_canada_boundary()
        
        empty_canada_raster = self.create_empty_raster(canada_boundary)
        
        target_dataset = self.burn_fire_polygons_to_raster(empty_canada_raster, fire_polygons_shp_path)
        
        target_file_path = self.save_target(target_dataset, year, self.raw_data_output_folder_path)
        
        shutil.rmtree(fire_polygons_shp_path.parent)
        
        return target_file_path

    def get_fire_polygons_path(self, year: int) -> Path:
        fire_polygons = self.fire_data_source.download(year)
        fire_polygons = fire_polygons.to_crs(epsg=self.target_epsg_code)
        
        fire_polygons_shp_path = self.raw_data_output_folder_path / "tmp"
        fire_polygons_shp_path.mkdir(parents=True, exist_ok=True)
        fire_polygons_shp_path = fire_polygons_shp_path / f"fire_occurrence_{year}.shp"
        fire_polygons.to_file(fire_polygons_shp_path, driver='ESRI Shapefile')
        
        return fire_polygons_shp_path

    def get_canada_boundary(self) -> gpd.GeoDataFrame:
        self.canada.load()
        canada_boundary = self.canada.boundary
        canada_boundary = canada_boundary.to_crs(epsg=self.target_epsg_code)
        return canada_boundary
    
    def create_empty_raster(self, canada_boundary: gpd.GeoDataFrame) -> gdal.Dataset:
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(self.target_epsg_code)
                
        minx, miny, maxx, maxy = canada_boundary.total_bounds
        width = int((maxx - minx) / self.resolution_in_meters)
        height = int((maxy - miny) / self.resolution_in_meters)
        
        mem_raster = gdal.GetDriverByName('MEM').Create('', width, height, 1, gdal.GDT_Byte)
        mem_raster.SetGeoTransform((minx, self.resolution_in_meters, 0, maxy, 0, -self.resolution_in_meters))
        mem_raster.SetProjection(target_srs.ExportToWkt())
        band = mem_raster.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.Fill(0)
        
        return mem_raster
    
    def burn_fire_polygons_to_raster(self, empty_raster: gdal.Dataset, fire_polygons_shp_path: Path) -> gdal.Dataset:
        fire_polygons_ds = gdal.OpenEx(str(fire_polygons_shp_path.resolve()))
        gdal.RasterizeLayer(empty_raster, [1], fire_polygons_ds.GetLayer(), burn_values=[1])
        
        return empty_raster
    
    def save_target(self, target_dataset: gdal.Dataset, year: int, base_path: Path) -> Path:
        driver = gdal.GetDriverByName('NetCDF')
        target_file_path = base_path / Path(f"{year}") / Path("fire_occurrences.nc")
        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        driver.CreateCopy(str(target_file_path.resolve()), target_dataset)
        
        return target_file_path
