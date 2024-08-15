from data_sources.canada_boundary_data_source import CanadaBoundaryDataSource
from pathlib import Path


province_name_to_pruid = {
    "NL": 10,
    "PE": 11,
    "NS": 12,
    "NB": 13,
    "QC": 24,
    "ON": 35,
    "MB": 46,
    "SK": 47,
    "AB": 48,
    "BC": 59,
    "YT": 60,
    "NT": 61,
    "NU": 62,
}


class CanadaBoundary:
    def __init__(self, data_source: CanadaBoundaryDataSource, target_epsg: int = 3978):
        self.data_source = data_source
        self.target_epsg = target_epsg
        self.province_name_to_pruid = province_name_to_pruid

    def load(self, provinces: list = list(province_name_to_pruid.keys())):
        self.boundary = self.data_source.download()
        selected_pruid = [
            str(self.province_name_to_pruid[province]) for province in provinces
        ]
        selected_pruid_mask = self.boundary["PRUID"].isin(selected_pruid)
        self.boundary = self.boundary[selected_pruid_mask]
        self.boundary = self.boundary.to_crs(epsg=self.target_epsg)

    def save(self, output_folder: Path) -> Path:
        boundary_folder = output_folder / Path("boundary")
        boundary_folder.mkdir(parents=True, exist_ok=True)
        self.boundary_file = boundary_folder / Path("boundary.shp")
        self.boundary.to_file(self.boundary_file, driver="ESRI Shapefile")
        return self.boundary_file
