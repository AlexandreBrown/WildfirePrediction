import time
import torch
import torch.nn as nn
import torchmetrics
from loguru import logger
from torch.utils.data import DataLoader
import torchmetrics.classification
import torchmetrics.segmentation
from datasets.wildfire_data_module import WildfireDataModule
from pathlib import Path


class SemanticSegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        data_module: WildfireDataModule,
        optimizer,
        device: torch.device,
        loss: nn.Module,
        optimization_metric_name: str,
        minimize_optimization_metric: bool,
        best_model_output_folder: Path,
    ):
        self.model = model.to(device)
        self.data_module = data_module
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        self.optimization_metric_name = optimization_metric_name
        self.minimize_optimization_metric = minimize_optimization_metric
        self.best_model_output_folder = best_model_output_folder
        self.best_model_output_folder.mkdir(parents=True, exist_ok=True)

        self.train_metrics = self.create_metrics()
        self.val_metrics = self.create_metrics()

        self.clear_best_model()
        logger.info("Trainer initialized!")

    def create_metrics(self) -> list:
        return [
            (
                "dice_coef",
                torchmetrics.segmentation.GeneralizedDiceScore(num_classes=1),
                self.metric_logits_to_class_index,
            ),
            (
                "pr_auc",
                torchmetrics.classification.BinaryAveragePrecision(),
                self.metric_logits_to_logits,
            ),
        ]

    def metric_logits_to_class_index(self, y_hat, y, metric):
        y_hat_binary = (torch.sigmoid(y_hat) > 0.5).long()
        return metric(y_hat_binary, y)

    def metric_logits_to_logits(self, y_hat, y, metric):
        return metric(y_hat, y)

    def train_model(
        self, max_nb_epochs: int, train_dl: DataLoader, val_dl: DataLoader
    ) -> dict:
        logger.info("Training model...")
        self.clear_best_model()
        self.train_step = 1
        self.epoch = 1
        self.train_dl = train_dl
        self.val_dl = val_dl

        start_time = time.time()
        for _ in range(max_nb_epochs):
            logger.info(f"Epoch {self.epoch}/{max_nb_epochs}")
            self.train_model_for_1_epoch()
            val_metrics = self.validate_model()
            self.update_best_model(val_metrics)
        end_time = time.time()

        train_duration_minutes = (end_time - start_time) / 60
        return {
            "train_duration_minutes": train_duration_minutes,
            "best_epoch": self.best_epoch,
            "best_train_step": self.best_train_step,
            "best_val_metric": self.best_val_metric,
            "best_model_path": self.best_model_path,
        }

    def clear_best_model(self):
        self.best_val_metric = (
            float("inf") if self.minimize_optimization_metric else -float("inf")
        )
        self.best_model_path = None
        self.best_epoch = None
        self.best_train_step = None

    def train_model_for_1_epoch(self):
        self.model.train()

        log_prefix = "train_"

        epoch_loss = 0.0
        number_of_batches = 0

        for X, y in self.train_dl:

            X = X.to(self.device)
            y = y.to(self.device)

            logger.debug(f"Predicting train batch X ({X.shape}) y ({y.shape})...")
            y_hat = self.model(X)
            y_hat = torch.squeeze(y_hat, dim=1)

            loss = self.loss(y_hat, y.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.log_step_metric(self.train_step, f"{log_prefix}loss", loss.item())

            for train_metric_name, train_metric, metric_wrapper in self.train_metrics:
                metric_value = metric_wrapper(y_hat, y, train_metric)

                self.log_step_metric(
                    self.train_step,
                    f"{log_prefix}{train_metric_name}",
                    metric_value.item(),
                )

            epoch_loss += loss.item()

            self.train_step += 1
            number_of_batches += 1

        epoch_loss /= number_of_batches
        self.log_epoch_metric(f"{log_prefix}loss", epoch_loss)

        for train_metric_name, train_metric, _ in self.train_metrics:
            metric_result = train_metric.compute()
            self.log_epoch_metric(
                f"{log_prefix}{train_metric_name}", metric_result.item()
            )

        self.epoch += 1

    def log_step_metric(self, step: int, metric_name: str, metric_value: float):
        logger.info(f"Step {step}: {metric_name}={metric_value}")

    def validate_model(self):
        self.model.eval()

        with torch.no_grad():
            log_prefix = "val_"

            val_metrics = {}

            val_loss = 0.0
            number_of_batches = 0

            for X, y in self.val_dl:

                X = X.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(X)
                y_hat = torch.squeeze(y_hat, dim=1)

                loss = self.loss(y_hat, y.float())

                val_loss += loss.item()
                number_of_batches += 1

                for val_metric_name, val_metric, metric_wrapper in self.val_metrics:
                    metric_wrapper(y_hat, y, val_metric)

            val_loss /= number_of_batches
            self.log_epoch_metric(f"{log_prefix}loss", val_loss)
            val_metrics[f"{log_prefix}loss"] = val_loss

            for val_metric_name, val_metric, _ in self.val_metrics:
                metric_result = val_metric.compute()
                self.log_epoch_metric(
                    f"{log_prefix}{val_metric_name}", metric_result.item()
                )
                val_metrics[val_metric_name] = metric_result.item()

        return val_metrics

    def log_epoch_metric(self, metric_name: str, metric_value: float):
        logger.info(f"Epoch {self.epoch}: {metric_name}={metric_value}")

    def update_best_model(self, val_metrics: dict):
        val_metric = val_metrics[self.optimization_metric_name]

        if self.minimize_optimization_metric:
            is_latest_val_metric_better = val_metric < self.best_val_metric
        else:
            is_latest_val_metric_better = val_metric > self.best_val_metric

        if is_latest_val_metric_better:
            self.best_val_metric = val_metric
            for file in self.best_model_output_folder.glob("*"):
                file.unlink()
            self.best_model_path = f"{self.best_model_output_folder}/best_model.pth"
            self.best_epoch = self.epoch
            self.best_train_step = self.train_step
            self.model.eval()
            torch.save(self.model.state_dict(), self.best_model_path)
