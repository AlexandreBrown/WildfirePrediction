import torch
import numpy as np
from torchvision import tv_tensors
from typing import Optional
from pathlib import Path
from torch.utils.data import Dataset
from osgeo import gdal
from raster_io.read import get_extension
from tensordict import TensorDict


class WildfireDataset(Dataset):
    def __init__(
        self,
        input_folder_path: Path,
        target_folder_path: Optional[Path] = None,
        input_data_indexes_to_remove: list = [],
        extension: str = get_extension("gtiff"),
        transform=None,
    ):
        gdal.UseExceptions()
        self.input_folder_path = input_folder_path
        self.target_folder_path = target_folder_path
        self.extension = extension
        self.transform = transform
        self.input_file_paths = self.get_file_paths(input_folder_path)
        self.target_file_paths = self.get_file_paths(target_folder_path)
        self.input_data_indexes_to_remove = input_data_indexes_to_remove

    def get_file_paths(self, folder_path: Optional[Path]) -> list:
        if folder_path is None:
            return []

        return sorted(list(folder_path.glob(f"*{self.extension}")))

    def __len__(self) -> int:
        return len(self.input_file_paths)

    def __getitem__(self, idx: int) -> TensorDict:
        input_dataset = gdal.Open(str(self.input_file_paths[idx]), gdal.GA_ReadOnly)
        input_data_numpy = np.delete(
            input_dataset.ReadAsArray(), self.input_data_indexes_to_remove, axis=0
        )

        input_data_img = tv_tensors.Image(
            torch.from_numpy(input_data_numpy), dtype=torch.float32, requires_grad=False
        )

        projection = input_dataset.GetProjection()
        geotransform = input_dataset.GetGeoTransform()

        del input_dataset

        if self.target_folder_path is None:
            if self.transform:
                input_data_img = self.transform(input_data_img)

            output = TensorDict(
                image=input_data_img,
                projection=projection,
                geotransform=geotransform,
            )
            return output

        target_dataset = gdal.Open(str(self.target_file_paths[idx]))
        target_data_numpy = target_dataset.ReadAsArray()
        target_mask = tv_tensors.Mask(
            torch.from_numpy(target_data_numpy), dtype=torch.long, requires_grad=False
        )

        del target_dataset

        if self.transform:
            input_data_img, target_mask = self.transform(input_data_img, target_mask)

        output = TensorDict(
            image=input_data_img,
            mask=target_mask,
            projection=projection,
            geotransform=geotransform,
        )

        return output
