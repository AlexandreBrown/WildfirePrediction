import requests
import zipfile
import io
import shutil
from pathlib import Path
from loguru import logger


class GovCanWaterBodiesDataSource:
    def __init__(self, output_path: Path):
        self.output_path = output_path

    def download(self) -> Path:
        download_output_folder = self.output_path / "gov_can_water_bodies"
        download_output_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Gov CAN Water Bodies...")
        download_request = requests.get(
            "https://ftp.geogratis.gc.ca/pub/nrcan_rncan/vector/framework_cadre/Atlas_of_Canada_1M/hydrology/AC_1M_Waterbodies.shp.zip"
        )
        zip = zipfile.ZipFile(io.BytesIO(download_request.content))
        zip.extractall(download_output_folder)
        files_to_move = []
        for file in (download_output_folder / "AC_1M_Waterbodies_shp").iterdir():
            if file.suffix in [".shp", ".shx", ".dbf", ".prj", ".shp.xml", ".xml"]:
                files_to_move.append(file)
        for file in files_to_move:
            shutil.move(file, download_output_folder)
        shutil.rmtree(download_output_folder / "AC_1M_Waterbodies_shp")
        shutil.rmtree(download_output_folder / "AC_1M_Etendues_d_eau_shp")

        return next(download_output_folder.glob("*.shp"))
