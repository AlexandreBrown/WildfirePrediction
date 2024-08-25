import torch
import subprocess
import multiprocessing as mp
import os
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
from transforms.replace_value_transform import ReplaceNanValueTransform
from datasets.wildfire_data import WildfireData
from datasets.collate import CollateDataAugs


class WildfireDataModule:
    def __init__(
        self,
        input_data_indexes_to_remove: list = [],
        seed: int = 42,
        train_batch_size: int = 4,
        eval_batch_size: int = 8,
        model_input_size: int = 256,
        destination_no_data_value: float = -32768.0,
        input_data_periods_folders_paths: Optional[list] = None,
        target_periods_folders_paths: Optional[list] = None,
        train_periods: Optional[list] = None,
        val_split: Optional[float] = None,
        preprocessing_num_workers: int = 4,
        output_folder_path: Optional[Path] = None,
        train_folder_path: Optional[Path] = None,
        val_folder_path: Optional[Path] = None,
        test_folder_path: Optional[Path] = None,
        predict_folder_path: Optional[Path] = None,
        train_stats: Optional[dict] = None,
        data_loading_num_workers: int = 4,
        device: Optional[torch.device] = None,
        data_augs: Optional[dict] = None,
    ):
        gdal.UseExceptions()
        self.input_data_indexes_to_remove = input_data_indexes_to_remove
        self.seed = seed
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.model_input_size = model_input_size
        self.preprocessing_num_workers = preprocessing_num_workers
        self.destination_no_data_value = destination_no_data_value
        self.train_stats = train_stats
        self.generator = torch.Generator().manual_seed(seed)
        self.output_folder_path = output_folder_path
        self.input_data_periods_folders_paths = input_data_periods_folders_paths
        self.target_periods_folders_paths = target_periods_folders_paths
        self.train_periods = train_periods
        self.val_split = val_split
        self.train_folder_path = train_folder_path
        self.val_folder_path = val_folder_path
        self.test_folder_path = test_folder_path
        self.predict_folder_path = predict_folder_path
        self.data_loading_num_workers = data_loading_num_workers
        self.device = device
        self.data_augs = data_augs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        self.input_folder_name = "input"
        self.target_folder_name = "target"

    def split_data(self) -> dict:
        assert self.input_data_periods_folders_paths is not None
        assert self.target_periods_folders_paths is not None
        assert self.train_periods is not None and len(self.train_periods) > 0
        assert self.output_folder_path is not None

        logger.info("Splitting data...")
        self.output_folder_path.mkdir(parents=True, exist_ok=True)
        self.val_folder_path = self.output_folder_path / "val"
        self.val_folder_path.mkdir(parents=True, exist_ok=True)
        self.train_folder_path = self.output_folder_path / "train"
        self.train_folder_path.mkdir(parents=True, exist_ok=True)
        self.test_folder_path = self.output_folder_path / "test"
        self.test_folder_path.mkdir(parents=True, exist_ok=True)

        (
            train_input_data_folders_paths,
            train_target_folders_paths,
            test_input_data_folders_paths,
            test_target_folders_paths,
        ) = self.get_train_test_folders_paths()

        logger.info(f"Train input data folders: {len(train_input_data_folders_paths)}")
        logger.info(f"Train target folders: {len(train_target_folders_paths)}")
        logger.info(f"Test input data folders: {len(test_input_data_folders_paths)}")
        logger.info(f"Test target folders: {len(test_target_folders_paths)}")

        train_input_files = self.get_files(train_input_data_folders_paths)
        logger.info(f"Train input files: {len(train_input_files)}")
        train_target_files = self.get_files(train_target_folders_paths)
        logger.info(f"Train target files: {len(train_target_files)}")

        test_input_files = self.get_files(test_input_data_folders_paths)
        logger.info(f"Test input files: {len(test_input_files)}")
        test_target_files = self.get_files(test_target_folders_paths)
        logger.info(f"Test target files: {len(test_target_files)}")

        if self.val_split is not None and self.val_split > 0.0:
            logger.info("Splitting train dataset into train and validation datasets...")
            (
                train_input_files,
                val_input_files,
                train_target_files,
                val_target_files,
            ) = train_test_split(
                sorted(train_input_files),
                sorted(train_target_files),
                test_size=self.val_split,
                random_state=self.seed,
            )
            logger.info(f"Train input files: {len(train_input_files)}")
            logger.info(f"Validation input files: {len(val_input_files)}")
            logger.info("Preprocessing validation files...")
            self.preprocess_tiles(
                self.val_folder_path, val_input_files, val_target_files
            )

        logger.info("Preprocessing train files...")
        self.preprocess_tiles(
            self.train_folder_path, train_input_files, train_target_files
        )

        logger.info("Computing training statistics...")
        stats = ImageStats()
        self.train_stats = stats.compute_aggregated_files_stats(train_input_files)
        logger.info("Computed training statistics!")

        logger.info("Preprocessing test files...")
        self.preprocess_tiles(
            self.test_folder_path, test_input_files, test_target_files
        )

        logger.success("Data split completed!")

        return {
            "train_folder_path": str(self.train_folder_path),
            "val_folder_path": str(self.val_folder_path),
            "test_folder_path": str(self.test_folder_path),
            "train_stats": self.train_stats,
        }

    def get_train_test_folders_paths(self) -> tuple:
        train_input_data_folders_paths = set()
        train_targets_folders_paths = set()
        test_input_data_folders_paths = set()
        test_targets_folders_paths = set()

        for input_data_folder_path, target_folder_path in zip(
            self.input_data_periods_folders_paths, self.target_periods_folders_paths
        ):
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

        return (
            train_input_data_folders_paths,
            train_targets_folders_paths,
            test_input_data_folders_paths,
            test_targets_folders_paths,
        )

    def preprocess_tiles(
        self, output_folder: Path, input_files: list, target_files: list
    ):
        input_data_folder = output_folder / self.input_folder_name
        input_data_folder.mkdir(parents=True)

        target_folder = output_folder / self.target_folder_name
        target_folder.mkdir()

        self.create_tiles_sized_for_model(
            input_files, input_data_folder, output_type="Float32"
        )
        self.create_tiles_sized_for_model(
            target_files, target_folder, output_type="Byte"
        )

    def create_tiles_sized_for_model(
        self, files: list, output_folder: Path, output_type: str
    ):
        if len(files) == 0:
            return

        tile_size_in_pixels = self.model_input_size

        args = [
            (file, output_folder, tile_size_in_pixels, output_type) for file in files
        ]

        nb_processes = min(
            min(
                max(1, len(os.sched_getaffinity(0)) - 1), self.preprocessing_num_workers
            ),
            len(files),
        )
        with mp.Pool(processes=nb_processes) as pool:
            pool.starmap(self.create_tile_sized_for_model, args)

    def create_tile_sized_for_model(
        self,
        file: Path,
        output_folder: Path,
        tile_size_in_pixels: int,
        output_type: str,
    ):
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
                tile_output_file_path = (
                    output_folder / f"{file.parent.stem}_{file.stem}_{x}_{y}{extension}"
                )

                cmd = [
                    "gdal_translate",
                    "-strict",
                    "-ot",
                    output_type,
                    "-srcwin",
                    str(pixels_x_offset),
                    str(pixels_y_offset),
                    str(tile_size_in_pixels),
                    str(tile_size_in_pixels),
                    str(file),
                    str(tile_output_file_path),
                ]

                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

    def get_files(self, folders: list) -> list:
        files = []

        for folder in folders:
            for file in folder.glob(f"*{get_extension('gtiff')}"):
                files.append(file)

        return sorted(files)

    def setup(self, stage: str):
        logger.info("Setting up data module...")

        if stage == "fit":
            logger.info("Creating train dataset...")
            train_transform = self.get_train_transform()
            self.train_dataset = self.create_dataset(
                self.train_folder_path, train_transform
            )

            logger.info("Creating validation dataset...")
            val_transform = self.get_eval_transform()
            self.val_dataset = self.create_dataset(self.val_folder_path, val_transform)

            logger.info("Creating test dataset...")
            test_transform = self.get_eval_transform()
            self.test_dataset = self.create_dataset(
                self.test_folder_path, test_transform
            )
        elif stage == "predict":
            logger.info("Creating predict dataset...")
            predict_transform = self.get_eval_transform()
            self.predict_dataset = self.create_dataset(
                self.predict_folder_path, predict_transform, has_target=False
            )

        logger.success("Data module setup completed!")

    def get_train_transform(self) -> v2.Compose:
        assert (
            self.train_stats is not None
        ), "Train stats must be computed before creating the training transform!"

        mean = self.train_stats["mean"]
        std = self.train_stats["std"]

        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
                ReplaceNanValueTransform(
                    replace_value=self.destination_no_data_value,
                ),
            ]
        )

    def get_eval_transform(self) -> v2.Compose:
        assert (
            self.train_stats is not None
        ), "Train stats must be computed before creating the evaluation transform!"
        mean = self.train_stats["mean"]
        std = self.train_stats["std"]

        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
                ReplaceNanValueTransform(
                    replace_value=self.destination_no_data_value,
                ),
            ]
        )

    def create_dataset(
        self,
        data_folder: Path,
        transform=None,
        has_target: bool = True,
    ):
        input_folder_path = data_folder / self.input_folder_name

        if has_target:
            target_folder = data_folder / self.target_folder_name
        else:
            target_folder = None

        dataset = WildfireDataset(
            input_folder_path=input_folder_path,
            target_folder_path=target_folder,
            input_data_indexes_to_remove=self.input_data_indexes_to_remove,
            transform=transform,
        )

        efficient_dataset = WildfireData.from_dataset(
            dataset,
            num_workers=self.data_loading_num_workers,
            batch=self.train_batch_size,
        )

        return efficient_dataset

    def train_dataloader(self):

        data_augs = self.create_data_augs()

        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=CollateDataAugs(data_augs=data_augs),
        )

    def create_data_augs(self) -> Optional[v2.Compose]:
        if self.data_augs is None or len(self.data_augs) == 0:
            return None

        data_augs = []

        for data_aug in self.data_augs:
            if data_aug["name"] == "RandomHorizontalFlip":
                data_augs.append(v2.RandomHorizontalFlip(**data_aug["params"]))
            elif data_aug["name"] == "RandomVerticalFlip":
                data_augs.append(v2.RandomVerticalFlip(**data_aug["params"]))
            else:
                raise ValueError(f"Unknown data augmentation: {data_aug['name']}")

        logger.info(f"Created {len(data_augs)} data augmentations!")

        return v2.Compose(data_augs)

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=CollateDataAugs(),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=CollateDataAugs(),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=CollateDataAugs(),
        )
