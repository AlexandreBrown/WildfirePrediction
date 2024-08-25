import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from datasets.wildfire_data_module import WildfireDataModule
from pathlib import Path
from loggers.factory import LoggerFactory
from metrics.pr_auc import PrecisionRecallAuc


class SemanticSegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        data_module: WildfireDataModule,
        optimizer,
        device: torch.device,
        loss: nn.Module,
        loss_name: str,
        optimization_metric_name: str,
        minimize_optimization_metric: bool,
        best_model_output_folder: Path,
        logger_factory: LoggerFactory,
        output_folder: Path,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        self.loss_name = loss_name
        self.optimization_metric_name = optimization_metric_name
        self.minimize_optimization_metric = minimize_optimization_metric
        self.best_model_output_folder = best_model_output_folder
        self.best_model_output_folder.mkdir(parents=True, exist_ok=True)
        self.logger_factory = logger_factory
        self.output_folder = output_folder
        self.train_metrics = self.create_metrics()
        self.val_metrics = self.create_metrics()
        self.test_metrics = self.create_metrics()
        self.clear_best_model()
        logger.info("Trainer initialized!")

    def create_metrics(self) -> list:
        return [PrecisionRecallAuc()]

    def train_model(self, max_nb_epochs: int, train_dl: DataLoader, val_dl: DataLoader):
        logger.info("Training model...")
        self.clear_best_model()
        self.train_step = 0
        self.epoch = 0
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_logger = self.logger_factory.create("train_")
        self.val_logger = self.logger_factory.create("val_")

        start_time = time.time()

        epochs = tqdm(range(max_nb_epochs), desc="Epochs")

        for _ in epochs:
            self.epoch += 1
            self.train_model_for_1_epoch()
            self.validate_model()
            self.update_epoch_metrics_progress(epochs)
            self.update_best_model()
            self.train_logger.on_epoch_end(self.epoch)
            self.val_logger.on_epoch_end(self.epoch)

        end_time = time.time()

        train_duration_in_minutes = (end_time - start_time) / 60

        train_results = {
            "train_duration_in_minutes": train_duration_in_minutes,
            "best_epoch": self.best_epoch,
            "best_train_step": self.best_train_step,
            "optimization_val_metric_name": self.optimization_metric_name,
            "best_val_metric_value": self.best_val_metric,
            "best_model_path": self.best_model_path,
        }

        self.save_training_results(train_results)

        logger.success("Training completed successfully!")

    def update_epoch_metrics_progress(self, epochs):
        formatted_train_metrics = self.train_logger.format_metrics(
            self.train_logger.epoch_metrics
        )
        formatted_train_metrics = {
            f"train_{k}": v for k, v in formatted_train_metrics.items()
        }

        formatted_val_metrics = self.val_logger.format_metrics(
            self.val_logger.epoch_metrics
        )
        formatted_val_metrics = {
            f"val_{k}": v for k, v in formatted_val_metrics.items()
        }

        epochs.set_postfix({**formatted_train_metrics, **formatted_val_metrics})

    def save_training_results(self, train_result: dict):
        logger.info(f"Training result: {train_result}")
        train_result_path = self.output_folder / "train_result.json"
        with open(train_result_path, "w") as f:
            json.dump(train_result, f, indent=4)
        logger.info(f"Training result saved to {str(train_result_path)}")

    def clear_best_model(self):
        self.best_val_metric = (
            float("inf") if self.minimize_optimization_metric else float("-inf")
        )
        self.best_model_path = None
        self.best_epoch = None
        self.best_train_step = None

    def train_model_for_1_epoch(self):
        self.model.train()

        epoch_loss = 0.0

        train_loader = tqdm(self.train_dl, desc="Training", leave=False)

        for train_data in train_loader:
            self.train_step += 1

            X = train_data.images
            y = train_data.masks

            logger.debug(f"Predicting train batch X ({X.shape}) y ({y.shape})...")
            y_hat = self.model(X)
            logger.debug(f"Predicted y_hat ({y_hat.shape})")
            y_hat = torch.squeeze(y_hat, dim=1)
            logger.debug(f"Predicted y_hat after squeeze ({y_hat.shape})")

            logger.debug("Computing loss...")
            logger.debug(f"y_hat {y_hat.shape} y {y.shape}")
            loss = self.loss(y_hat, y.float())

            logger.debug("Optimizing model...")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_logger.log_step_metric(self.loss_name, loss.item())

            for train_metric in self.train_metrics:
                metric_value = train_metric(y_hat, y)
                self.train_logger.log_step_metric(
                    train_metric.name, metric_value.item()
                )

            formatted_metrics = self.train_logger.format_metrics(
                self.train_logger.step_metrics
            )
            train_loader.set_postfix(formatted_metrics)
            self.train_logger.on_step_end(self.train_step)

            epoch_loss += loss.item()

        epoch_loss /= len(self.train_dl)
        self.train_logger.log_epoch_metric(self.loss_name, epoch_loss)

        for train_metric in self.train_metrics:
            metric_result = train_metric.compute()
            self.train_logger.log_epoch_metric(train_metric.name, metric_result.item())

    def validate_model(self):
        logger.debug("Validating model...")
        self.model.eval()

        val_loss = 0.0

        val_loader = tqdm(self.val_dl, desc="Validation", leave=False)

        with torch.no_grad():
            for val_data in val_loader:

                X = val_data.images
                y = val_data.masks

                y_hat = self.model(X)
                y_hat = torch.squeeze(y_hat, dim=1)

                loss = self.loss(y_hat, y.float())

                val_loss += loss.item()

                for val_metric in self.val_metrics:
                    val_metric(y_hat, y)

            val_loss /= len(self.val_dl)
            self.val_logger.log_epoch_metric(self.loss_name, val_loss)

            for val_metric in self.val_metrics:
                metric_result = val_metric.compute()
                self.val_logger.log_epoch_metric(val_metric.name, metric_result.item())

    def update_best_model(self):
        logger.debug("Updating best model...")
        val_metric_to_optimize = self.val_logger.epoch_metrics[
            self.optimization_metric_name
        ]

        if self.minimize_optimization_metric:
            is_model_better = val_metric_to_optimize < self.best_val_metric
        else:
            is_model_better = val_metric_to_optimize > self.best_val_metric

        logger.info(
            f"is_model_better {is_model_better} | Previous {self.optimization_metric_name} : {self.best_val_metric} | new {self.optimization_metric_name} : {val_metric_to_optimize}"
        )

        if is_model_better:
            self.best_val_metric = val_metric_to_optimize
            for file in self.best_model_output_folder.glob("*"):
                file.unlink()
            self.best_model_path = f"{self.best_model_output_folder}/best_model_epoch_{self.epoch}_step_{self.train_step}.pth"
            self.best_epoch = self.epoch
            self.best_train_step = self.train_step
            self.model.eval()
            torch.save(self.model.state_dict(), self.best_model_path)

    def test_model(self, test_dl: DataLoader):
        logger.info("Testing model...")
        assert (
            self.train_dl != test_dl and self.val_dl != test_dl
        ), "Test set should be different from train and val sets!"

        best_trained_model = self.model.load_state_dict(
            torch.load(self.best_model_path)
        ).to(self.device)

        test_loss = 0.0
        self.test_dl = test_dl
        self.test_logger = self.logger_factory.create("test_")

        best_trained_model.eval()

        test_loader = tqdm(self.test_dl, desc="Testing", leave=False)

        for test_data in test_loader:

            X = test_data.images
            y = test_data.masks

            y_hat = best_trained_model(X)
            y_hat = torch.squeeze(y_hat, dim=1)

            loss = self.loss(y_hat, y.float())

            test_loss += loss.item()

            for test_metric in self.test_metrics:
                test_metric(y_hat, y)

        test_loss /= len(self.test_dl)
        self.test_logger.log_epoch_metric(self.loss_name, test_loss)

        for test_metric in self.test_metrics:
            metric_result = test_metric.compute()
            self.test_logger.log_epoch_metric(test_metric.name, metric_result.item())

        self.test_logger.on_epoch_end(epoch=1)

        logger.success("Testing completed successfully!")
