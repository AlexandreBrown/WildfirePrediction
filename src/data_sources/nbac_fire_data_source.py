import requests
import zipfile
import io
from pathlib import Path
import geopandas as gpd


class NbacFireDataSource:
    def __init__(self, raw_data_path: Path = Path("../data/raw/")):
        self.raw_data_path = raw_data_path
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
    
    def download(self, year: int) -> gpd.GeoDataFrame:
        
        output_path = self.raw_data_path / Path(f"{year}") / Path("nbac_fire_polygons/")
        output_path.mkdir(parents=True, exist_ok=True)
        
        if len(list(output_path.glob("./*.shp"))) == 0:
            print(f"Downloading NBAC fire polygons data for year {year}...")
            download_request = requests.get(f"https://cwfis.cfs.nrcan.gc.ca/downloads/nbac/nbac_{year}_20240530.zip")
            zip = zipfile.ZipFile(io.BytesIO(download_request.content))
            zip.extractall(output_path)
            for file in output_path.iterdir():
                if file.suffix == ".pdf" or file.suffix == ".html":
                    file.unlink()   
        else:
            print(f"Canada fire polygons already downloaded for year {year}, skipping download!")
            
        fire_polygons_shapefile_path = next(output_path.glob("./*.shp"))
        fire_polygons = gpd.read_file(fire_polygons_shapefile_path)
        
        return fire_polygons
        