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
from metrics.metric_factory import create_metrics
from losses.loss_factory import create_loss
from losses.nan_aware_loss import NanAwareLoss


class SemanticSegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        data_module: WildfireDataModule,
        optimizer,
        lr_scheduler,
        device: torch.device,
        loss_name: str,
        optimization_metric_name: str,
        minimize_optimization_metric: bool,
        best_model_output_folder: Path,
        logger_factory: LoggerFactory,
        output_folder: Path,
        metrics_config: list,
        target_no_data_value: int,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.loss_name = loss_name
        self.optimization_metric_name = optimization_metric_name
        self.minimize_optimization_metric = minimize_optimization_metric
        self.best_model_output_folder = best_model_output_folder
        self.best_model_output_folder.mkdir(parents=True, exist_ok=True)
        self.logger_factory = logger_factory
        self.output_folder = output_folder
        loss_config = self.get_loss_config(metrics_config, loss_name)
        self.loss = NanAwareLoss(
            create_loss(loss_name, **loss_config["params"]), target_no_data_value
        )
        self.train_metrics = create_metrics(metrics_config, target_no_data_value)
        self.val_metrics = create_metrics(metrics_config, target_no_data_value)
        self.test_metrics = create_metrics(metrics_config, target_no_data_value)
        self.clear_best_model()
        logger.info("Trainer initialized!")

    def get_loss_config(self, metrics_config: list, loss_name: str) -> dict:
        for metric in metrics_config:
            if metric["name"] == loss_name:
                return metric

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
            current_lr = self.get_lr()
            self.train_logger.log_epoch_metric("lr", current_lr)
            self.train_model_for_1_epoch()
            self.validate_model()
            train_epoch_metrics = self.train_logger.on_epoch_end(self.epoch)
            val_epoch_metrics = self.val_logger.on_epoch_end(self.epoch)
            self.update_epoch_metrics_progress(
                epochs, train_epoch_metrics, val_epoch_metrics
            )
            self.update_best_model(val_epoch_metrics)

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

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    def update_epoch_metrics_progress(
        self, epochs, train_epoch_metrics, val_epoch_metrics
    ):
        formatted_train_metrics = self.train_logger.format_metrics(train_epoch_metrics)
        formatted_train_metrics = {
            f"train_{k}": v for k, v in formatted_train_metrics.items()
        }

        formatted_val_metrics = self.val_logger.format_metrics(val_epoch_metrics)
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

        train_loader = tqdm(self.train_dl, desc="Training", leave=False)

        for train_data in train_loader:
            self.train_step += 1

            X = train_data.images.to(self.device)
            y = train_data.masks.to(self.device)

            y_hat = self.model(X)
            y_hat = torch.squeeze(y_hat, dim=1)

            loss = self.loss(y_hat, y.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            for train_metric in self.train_metrics:
                metric_value = train_metric(y_hat, y)
                self.train_logger.log_step_metric(train_metric.name, metric_value)

            train_step_metrics = self.train_logger.on_step_end(self.train_step)
            formatted_metrics = self.train_logger.format_metrics(train_step_metrics)
            train_loader.set_postfix(formatted_metrics)

            del X
            del y
            del y_hat
            del loss

        for train_metric in self.train_metrics:
            metric_result = train_metric.aggregate()
            self.train_logger.log_epoch_metric(train_metric.name, metric_result)

    def validate_model(self):
        logger.debug("Validating model...")
        self.model.eval()

        val_loader = tqdm(self.val_dl, desc="Validation", leave=False)

        with torch.no_grad():
            for val_data in val_loader:

                X = val_data.images.to(self.device)
                y = val_data.masks.to(self.device)

                y_hat = self.model(X)
                y_hat = torch.squeeze(y_hat, dim=1)

                loss = self.loss(y_hat, y.float())

                for val_metric in self.val_metrics:
                    val_metric(y_hat, y)

                del X
                del y
                del y_hat
                del loss

            for val_metric in self.val_metrics:
                metric_result = val_metric.aggregate()
                self.val_logger.log_epoch_metric(val_metric.name, metric_result)

    def update_best_model(self, val_epoch_metrics: dict):
        logger.debug("Updating best model...")
        val_metric_to_optimize = val_epoch_metrics[self.optimization_metric_name]

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

        self.model.load_state_dict(
            torch.load(self.best_model_path, weights_only=True), strict=True
        )
        self.model.to(self.device)

        self.test_dl = test_dl
        self.test_logger = self.logger_factory.create("test_")

        self.model.eval()

        test_loader = tqdm(self.test_dl, desc="Testing", leave=False)

        with torch.no_grad():
            for test_data in test_loader:

                X = test_data.images.to(self.device)
                y = test_data.masks.to(self.device)

                y_hat = self.model(X)
                y_hat = torch.squeeze(y_hat, dim=1)

                loss = self.loss(y_hat, y.float())

                for test_metric in self.test_metrics:
                    test_metric(y_hat, y)

                del X
                del y
                del y_hat
                del loss

        for test_metric in self.test_metrics:
            metric_result = test_metric.aggregate()
            self.test_logger.log_epoch_metric(test_metric.name, metric_result)

        self.test_logger.on_epoch_end(epoch=1)

        logger.success("Testing completed successfully!")
