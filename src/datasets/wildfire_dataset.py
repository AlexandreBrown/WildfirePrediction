import torch
from torchvision import tv_tensors
from typing import Optional
from pathlib import Path
from torch.utils.data import Dataset
from osgeo import gdal
from raster_io.read import get_extension


class WildfireDataset(Dataset):
    def __init__(
        self,
        input_folder_path: Path,
        target_folder_path: Optional[Path] = None,
        extension: str = get_extension("gtiff"),
        transform = None,
    ):
        gdal.UseExceptions()
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
        self.extension = extension
        self.transform = transform
        self.input_file_paths = self.get_file_paths(input_folder_path)
        self.target_file_paths = self.get_file_paths(target_folder_path)

    def get_file_paths(self, folder_path: Optional[Path]) -> list:
        if folder_path is None:
            return []

        return sorted(list(folder_path.glob(f"*{self.extension}")))

    def __len__(self) -> int:
        return len(self.input_file_paths)

    def __getitem__(self, idx: int) -> tuple:
        input_ds = gdal.Open(str(self.input_file_paths[idx]), gdal.GA_ReadOnly)
        input_data_numpy = input_ds.ReadAsArray()
        input_data = tv_tensors.Image(torch.from_numpy(input_data_numpy), dtype=torch.float32, requires_grad=False)

        del input_ds

        if self.target_folder_path is None:
            return input_data
        
        target_ds = gdal.Open(str(self.target_file_paths[idx]))
        target_data_numpy = target_ds.ReadAsArray()
        target_data = tv_tensors.Mask(torch.from_numpy(target_data_numpy), dtype=torch.long, requires_grad=False)

        del target_ds

        if self.transform:
            input_data, target_data = self.transform(input_data, target_data)

        return input_data, target_data
