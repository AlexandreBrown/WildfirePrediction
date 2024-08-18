import torch
import subprocess
import multiprocessing as mp
from loguru import logger
from osgeo import gdal
from pathlib import Path
from typing import Optional
from datasets.wildfire_dataset import WildfireDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from raster_io.read import get_extension
from stats.image import ImageStats
from torchvision.transforms import v2
from transforms.replace_value_transform import ReplaceValueTransform 


class WildfireDataModule:
    def __init__(
        self, 
        input_data_periods_folders_paths: list,
        input_data_indexes_to_remove: list,
        target_periods_folders_paths: list,
        train_periods: list,
        output_folder_path: Path,
        seed: int = 42,
        train_batch_size: int = 4,
        eval_batch_size: int = 8,
        val_split: Optional[float] = None,
        model_input_size: int = 256,
        num_workers: int = 4,
        source_no_data_value: float = -32768.0,
        destination_no_data_value: float = -32768.0,
        train_stats: Optional[dict] = None
    ):
        self.input_data_periods_folders_paths = input_data_periods_folders_paths
        self.input_data_indexes_to_remove = input_data_indexes_to_remove
        self.target_periods_folders_paths = target_periods_folders_paths
        self.train_periods = train_periods
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.val_split = val_split
        self.model_input_size = model_input_size
        self.num_workers = num_workers
        self.source_no_data_value = source_no_data_value
        self.destination_no_data_value = destination_no_data_value
        self.train_stats = train_stats
        self.generator = torch.Generator().manual_seed(seed)
        self.output_folder_path = output_folder_path
        self.output_folder_path.mkdir(parents=True, exist_ok=True)

    def setup(self, stage: str):
        logger.info(f"Setting up data module [Stage={stage}]...")
        
        train_input_data_folders_paths, train_target_folders_paths, test_input_data_folders_paths, test_target_folders_paths = self.get_train_test_folders_paths()
        
        logger.info(f"Train input data folders: {len(train_input_data_folders_paths)}")
        logger.info(f"Train target folders: {len(train_target_folders_paths)}")
        logger.info(f"Test input data folders: {len(test_input_data_folders_paths)}")
        logger.info(f"Test target folders: {len(test_target_folders_paths)}")
        
        if stage == "fit":
            train_input_files = self.get_files(train_input_data_folders_paths)
            logger.info(f"Train input files: {len(train_input_files)}")
            train_target_files = self.get_files(train_target_folders_paths)
            logger.info(f"Train target files: {len(train_target_files)}")
            
            stats = ImageStats()
            logger.info("Computing training statistics...")
            self.train_stats = stats.compute_aggregated_files_stats(train_input_files)
            logger.info("Computed training statistics!")
            
            if self.val_split is not None and self.val_split > 0.:
                logger.info("Splitting train dataset into train and validation datasets...")
                train_input_files, val_input_files, train_target_files, val_target_files = train_test_split(sorted(train_input_files), sorted(train_target_files), test_size=self.val_split, random_state=self.seed)
                logger.info(f"Train input files: {len(train_input_files)}")
                logger.info(f"Validation input files: {len(val_input_files)}")
                logger.info("Creating validation dataset...")
                val_transform = self.get_eval_transform()
                self.val_dataset = self.create_dataset(val_input_files, val_target_files, output_folder_name="val", transform=val_transform)
            
            logger.info("Creating train dataset...")
            train_transform = self.get_train_transform()
            self.train_dataset = self.create_dataset(train_input_files, train_target_files, output_folder_name="train", transform=train_transform)
  
        test_input_files = self.get_files(test_input_data_folders_paths)
        test_target_files = self.get_files(test_target_folders_paths)
        if len(test_input_files) == 0:
            return
        
        logger.info(f"Test input files: {len(test_input_files)}")
        logger.info("Creating test dataset...")
        test_transform = self.get_eval_transform()
        self.test_dataset = self.create_dataset(test_input_files, test_target_files, output_folder_name="test", transform=test_transform)

        logger.success(f"Data module setup completed [Stage={stage}]!")
        
    def get_train_test_folders_paths(self) -> tuple:
        train_input_data_folders_paths = set()
        train_targets_folders_paths = set()
        test_input_data_folders_paths = set()
        test_targets_folders_paths = set()

        for input_data_folder_path, target_folder_path in zip(self.input_data_periods_folders_paths, self.target_periods_folders_paths):
            assert input_data_folder_path.stem == target_folder_path.stem
            
            is_train_file = False
            
            for train_period in self.train_periods:
                train_period_folder_name = f"{train_period[0]}_{train_period[-1]}"
                
                if input_data_folder_path.stem == train_period_folder_name:
                    train_input_data_folders_paths.add(input_data_folder_path)
                    train_targets_folders_paths.add(target_folder_path)
                    is_train_file = True
                    break
                
            if not is_train_file:
                test_input_data_folders_paths.add(input_data_folder_path)
                test_targets_folders_paths.add(target_folder_path)
        
        return train_input_data_folders_paths, train_targets_folders_paths, test_input_data_folders_paths, test_targets_folders_paths
    
    def get_files(self, folders: list) -> list:
        files = []
        
        for folder in folders:
            for file in folder.glob(f"*{get_extension('gtiff')}"):
                files.append(file)
                
        return sorted(files)
    
    def get_train_transform(self) -> v2.Compose:
        mean = self.train_stats['mean']
        std = self.train_stats['std']
        
        return v2.Compose([
            ReplaceValueTransform(value_to_replace=self.source_no_data_value, replace_value=self.destination_no_data_value),
            v2.Normalize(mean=mean, std=std)
        ])
    
    def get_eval_transform(self) -> v2.Compose:
        assert self.train_stats is not None, "Train stats must be computed before creating the evaluation transform!"
        mean = self.train_stats['mean']
        std = self.train_stats['std']
        
        return v2.Compose([
            ReplaceValueTransform(value_to_replace=self.source_no_data_value, replace_value=self.destination_no_data_value),
            v2.Normalize(mean=mean, std=std)
        ])
    
    def create_dataset(self, input_files: list, target_files: list, output_folder_name: str, transform=None):
        dataset_output_folder = self.output_folder_path / output_folder_name
        dataset_output_folder.mkdir(parents=True)
        
        input_data_folder = dataset_output_folder / "input"
        input_data_folder.mkdir()
        
        target_folder = dataset_output_folder / "target"
        target_folder.mkdir()
        
        self.create_tiles_sized_for_model(input_files, input_data_folder)
        self.create_tiles_sized_for_model(target_files, target_folder)
        
        return WildfireDataset(
            input_folder_path=input_data_folder,
            target_folder_path=target_folder,
            input_data_indexes_to_remove=self.input_data_indexes_to_remove,
            transform=transform
        )
    
    def create_tiles_sized_for_model(self, files: list, output_folder: Path):
        tile_size_in_pixels = self.model_input_size
        
        args = [(file, output_folder, tile_size_in_pixels) for file in files]
        
        nb_processes = min(min(mp.cpu_count(), self.num_workers), len(files))
        with mp.Pool(processes=nb_processes) as pool:
            pool.starmap(self.create_tile_sized_for_model, args)
    
    def create_tile_sized_for_model(self, file: Path, output_folder: Path, tile_size_in_pixels: int):
        dataset = gdal.Open(str(file), gdal.GA_ReadOnly)
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        
        number_of_tiles_horizontally = width // tile_size_in_pixels
        number_of_tiles_vertically = height // tile_size_in_pixels
        
        for x in range(number_of_tiles_horizontally):
            for y in range(number_of_tiles_vertically):
                pixels_x_offset = x * tile_size_in_pixels
                pixels_y_offset = y * tile_size_in_pixels
                extension = get_extension("gtiff")
                tile_output_file_path = output_folder / f"{file.stem}_{x}_{y}{extension}"
                
                cmd = [
                    'gdal_translate',
                    '-strict',
                    '-ot',
                    'Float32',
                    '-srcwin', 
                    str(pixels_x_offset), 
                    str(pixels_y_offset), 
                    str(tile_size_in_pixels), 
                    str(tile_size_in_pixels),
                    str(file),
                    str(tile_output_file_path)
                ]
                
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)
