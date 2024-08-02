from typing import Optional
import torch
from pathlib import Path
from torch.utils.data import Dataset
from osgeo import gdal 


class WildfireDataset(Dataset):
    def __init__(
        self, 
        input_folder_path: Path,
        target_folder_path: Optional[Path] = None,
        extension: str='nc', 
        transform=None
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
        
        return sorted(list(folder_path.glob(f"*.{self.extension}")))
    
    def __len__(self) -> int:
        return len(self.input_file_paths)
    
    def __getitem__(self, idx: int) -> tuple:
        input_ds = gdal.Open(str(self.input_file_paths[idx]))
        input_data = torch.from_numpy(input_ds.GetRasterBand(1).ReadAsArray())
        
        target_ds = gdal.Open(str(self.target_file_paths[idx]))
        target_data = torch.from_numpy(target_ds.GetRasterBand(1).ReadAsArray())
        
        if self.transform:
            input_data, target_data = self.transform(input_data, target_data)
        
        return input_data, target_data
        