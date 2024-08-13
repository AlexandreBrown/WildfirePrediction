from typing import Optional
from pathlib import Path
from osgeo import gdal


def get_formatted_file_path(file_path: Path, netcdf_layer: Optional[str] = None) -> str:
    absolute_file_path = str(file_path.resolve())

    if file_path.suffix == ".nc":
        path = f"NETCDF:\"{absolute_file_path}\"{':' + netcdf_layer if netcdf_layer != '' else ''}"
    else:
        path = absolute_file_path

    return path


def open_dataset(
    file_path: Path, netcdf_layer: Optional[str] = None, read_only: bool = True
) -> gdal.Dataset:
    path = get_formatted_file_path(file_path, netcdf_layer)
    read_flag = gdal.GA_ReadOnly if read_only else gdal.GA_Update
    return gdal.Open(path, read_flag)


def get_extension(format: str) -> str:
    if format.lower() == "gtiff":
        extension = ".tiff"
    elif format.lower() == "netcdf":
        extension = ".nc"
    else:
        raise ValueError(f"Unknown format: {format}")

    return extension
