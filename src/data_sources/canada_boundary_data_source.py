import requests
import zipfile
import io
from pathlib import Path
import geopandas as gpd

class CanadaBoundaryDataSource:
    def __init__(self, output_path : Path = Path("../data/canada_boundary/")):
        self.output_path = output_path
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def download(self) -> gpd.GeoDataFrame:
        if len(list(self.output_path.glob("./*.shp"))) == 0:
            print("Downloading canada boundary shapefile...")
            download_request = requests.get("https://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/files-fichiers/2016/lpr_000b16a_e.zip")
            zip = zipfile.ZipFile(io.BytesIO(download_request.content))
            zip.extractall(self.output_path)
            for file in self.output_path.iterdir():
                if file.suffix == ".pdf" or file.suffix == ".html":
                    file.unlink()
        else:
            print("Canada boundary shapefile already downloaded, skipping download!")
        
        boundary_shapefile_path = next(self.output_path.glob("./*.shp"))
        return gpd.read_file(boundary_shapefile_path)
