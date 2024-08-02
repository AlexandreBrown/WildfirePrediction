import torch
from pathlib import Path
from random import shuffle
from typing import Optional
from datasets.wildfire_dataset import WildfireDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from targets.fire_occurrence_target import FireOccurrenceTarget


class WildfireDataModule:
    def __init__(
        self, 
        input_data_folder_path: Path,
        input_data_period: range,
        input_data_names: list,
        extension: str = 'nc',
        seed: int = 42,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        test_split: float = 0.2,
        val_split: Optional[float] = None,
        target: Optional[FireOccurrenceTarget] = None,
        target_data_period: Optional[range] = None
    ):
        self.input_data_folder_path = input_data_folder_path
        self.input_data_period = input_data_period
        self.input_data_names = input_data_names
        self.extension = extension
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_split = test_split
        self.val_split = val_split
        self.target = target
        self.target_data_period = target_data_period
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)
    
    def setup(self, stage: str):
        
        # TODO
        # 1. Loop self.input_data_folder_path for each year in input_data_period then keep path of data under self.input_data_folder_path / year / name for all name that are in self.input_data_names
        self.input_data_file_paths = []
        
        
        if self.target is not None:
            self.target_filepaths = self.target.generate_target_for_years(self.target_data_period)
            self.target_folder_path = self.target_filepaths[0].parent
        else:
            self.target_folder_path = None
        
        dataset_full = WildfireDataset(self.input_data_folder_path, target_folder_path=self.target_folder_path, extension=self.extension)
        
        self.train_dataset, self.test_dataset = random_split(dataset_full, [1 - self.test_split, self.test_split], generator=self.generator)
        if self.val_split is not None:
            self.train_dataset, self.val_dataset = random_split(self.train_dataset, [1 - self.val_split, self.val_split], generator=self.generator)

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False)
