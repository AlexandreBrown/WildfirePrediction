import geopandas as gpd
from pathlib import Path
from boundaries.canada_boundary import CanadaBoundary
from data_sources.nbac_fire_data_source import NbacFireDataSource
from osgeo import gdal, osr


class FireOccurrenceTarget:
    def __init__(
        self, 
        fire_data_source: NbacFireDataSource,
        boundary: CanadaBoundary,
        target_pixel_size_in_meters: int,
        target_epsg_code: int = 3978,
        output_folder_path: Path = Path("../data/target/")
    ):
        self.fire_data_source = fire_data_source
        self.boundary = boundary
        self.target_pixel_size_in_meters = target_pixel_size_in_meters
        self.target_epsg_code = target_epsg_code
        self.output_folder_path = output_folder_path
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
    
    def generate_target_for_years_ranges(self, years_ranges: list) -> dict:
        years_ranges_output_paths = {}
        
        for years_range in years_ranges:
            years_ranges_output_paths[(years_range[0], years_range[-1])] = self.generate_target_for_years(years_range, years_ranges_output_paths)
        
        return years_ranges_output_paths
        
    
    def generate_target_for_years(self, years: range) -> Path:
        years_fire_polygons = self.get_years_fire_polygons(years)
        
        years_fire_polygon_shapefile_path = self.output_folder_path / Path(f"fire_polygons_{years[0]}_{years[-1]}.shp")
        years_fire_polygons.to_file(years_fire_polygon_shapefile_path, driver='ESRI Shapefile')
        
        output_dataset_path = self.output_folder_path / Path(f"target_{years[0]}_{years[-1]}.nc")
        
        boundary_extent = self.boundary.boundary.total_bounds
        x_min, y_min, x_max, y_max = boundary_extent
        
        x_res = int((x_max - x_min) / self.target_pixel_size_in_meters)
        y_res = int((y_max - y_min) / self.target_pixel_size_in_meters)
        
        driver = gdal.GetDriverByName('NetCDF')
        output_target_dataset = driver.Create(str(output_dataset_path), x_res, y_res, 1, gdal.GDT_Byte)
        output_target_dataset.SetGeoTransform((x_min, self.target_pixel_size_in_meters, 0, y_max, 0, -self.target_pixel_size_in_meters))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(self.target_epsg_code)
        output_target_dataset.SetProjection(srs.ExportToWkt())
        
        band = output_target_dataset.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.Fill(0)
        
        years_fire_polygons_shapefile = gdal.OpenEx(str(years_fire_polygon_shapefile_path), gdal.OF_VECTOR)
        years_fire_polygons_layer = years_fire_polygons_shapefile.GetLayer()
        bands_to_rasterize = [1]
        gdal.RasterizeLayer(output_target_dataset, bands_to_rasterize, years_fire_polygons_layer, burn_values=[1])
        
        output_target_dataset.FlushCache()
        
        return output_dataset_path
    
    def get_years_fire_polygons(self, years: range) -> gpd.GeoDataFrame:
        years_fire_polygons = None
        
        for year in years:
            year_fire_polygons = self.fire_data_source.download(year)
            year_fire_polygons = year_fire_polygons.to_crs(epsg=self.target_epsg_code)
            year_fire_polygons = self.boundary.boundary.intersection(year_fire_polygons)
            
            if years_fire_polygons is None:
                years_fire_polygons = year_fire_polygons
            else:
                years_fire_polygons = gpd.overlay(years_fire_polygons, year_fire_polygons, how='union')
        
        return years_fire_polygons
