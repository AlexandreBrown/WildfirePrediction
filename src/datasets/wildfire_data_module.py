import torch
from pathlib import Path
from typing import Optional
from datasets.wildfire_dataset import WildfireDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from raster_io.read import get_extension


class WildfireDataModule:
    def __init__(
        self, 
        input_data_periods_folders_paths: list,
        input_data_indexes_to_keep: list,
        target_periods_folders_paths: list,
        train_periods: list,
        output_folder_path: Path,
        extension: str = get_extension("gtiff"),
        seed: int = 42,
        train_batch_size: int = 8,
        eval_batch_size: int = 16,
        val_split: Optional[float] = None,
        model_input_size: int = 256,
        num_workers: int = 4,
    ):
        self.input_data_periods_folders_paths = input_data_periods_folders_paths
        self.input_data_indexes_to_keep = input_data_indexes_to_keep
        self.target_periods_folders_paths = target_periods_folders_paths
        self.train_periods = train_periods
        self.extension = extension
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_split = val_split
        self.model_input_size = model_input_size
        self.num_workers = num_workers
        self.generator = torch.Generator().manual_seed(seed)
    
    def setup(self, stage: str):
        # 1. find all folder paths for train periods (input_data_periods_folders_paths), set test periods by checking the periods of the remaining folders
        
        # 2. Create symlink for data from train periods folder paths to output_folder_path / train
        # 3  if val_split is not None: Split train dataset into train and validation symlink paths
        # 4. For each tile in train folder, create tiles of size model_input_size x model_input_size
        # 5. For each tile in validation folder, create tiles of size model_input_size x model_input_size
        # 6. Create train WildfireDataset from train folder
        # 7. Create validation WildfireDataset from validation folder
        # 8. if teset folder paths is not empty : Create symlink for data from test periods folder paths to output_folder_path / test
        # 9. For each tile in test folder, create tiles of size model_input_size x model_input_size
        # 10. Create test WildfireDataset from test folder            
        
        pass    
        # train_dataset = WildfireDataset()
        # if self.val_split is not None:
        #     self.train_dataset, self.val_dataset = random_split(self.train_dataset, [1 - self.val_split, self.val_split], generator=self.generator)

    def get_train_test_input_data_folders_paths(self):
        pass
        # train_input_data_folders_paths = []
        # test_input_data_folders_paths = []

        # for input_data_period_folder_path in input_data_periods_folders_paths:
        #     print(f"input_data_period_folder_path: {input_data_period_folder_path}")
        #     is_train_period = False
        #     folder_path = Path(input_data_period_folder_path)
            
        #     for train_period in train_periods:
        #         train_period_folder_name = f"{train_period[0]}_{train_period[-1]}"
                
        #         print(f"train_period_folder_name: {train_period_folder_name}")
        #         if folder_path.stem == train_period_folder_name:
        #             print(f"MATCH FOUND! {input_data_period_folder_path} == {train_period_folder_name}")
        #             train_input_data_folders_paths.append(folder_path)
        #             is_train_period = True
        #             break
        #     if not is_train_period:
        #         test_input_data_folders_paths.append(folder_path)
            
        #     print("")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)
