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
        target_epsg_code: int = 3347,
        output_folder_path: Path = Path("../data/raw/")
    ):
        self.fire_data_source = fire_data_source
        self.canada = canada
        self.resolution_in_meters = resolution_in_meters
        self.target_epsg_code = target_epsg_code
        self.output_folder_path = output_folder_path
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
    
    def generate(self, year: int) -> Path:
        fire_polygons = self.fire_data_source.download(year)
        fire_polygons = fire_polygons.to_crs(epsg=self.target_epsg_code)
        fire_polygons_shp_path = self.output_folder_path / f"fire_occurrence_{year}.shp"
        fire_polygons.to_file(fire_polygons_shp_path, driver='ESRI Shapefile')
        
        self.canada.load()
        canada_boundary = self.canada.boundary
        canada_boundary = canada_boundary.to_crs(epsg=self.target_epsg_code)
        
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
        
        fire_polygons_ds = gdal.OpenEx(str(fire_polygons_shp_path.resolve()))
        gdal.RasterizeLayer(mem_raster, [1], fire_polygons_ds.GetLayer(), burn_values=[1])
        
        driver = gdal.GetDriverByName('NetCDF')
        target_file_path = self.output_folder_path / Path(f"{year}") / Path("fire_occurrences.nc")
        target_file_path.parent.mkdir(parents=True, exist_ok=True)
        driver.CreateCopy(str(target_file_path.resolve()), mem_raster)
        
        fire_polygons_shp_path.unlink()
        
        print(f"Successfully created target for year {year}!")
        
        return target_file_path
