import torch
from abc import ABC, abstractmethod
from loguru import logger


class Metric(ABC):
    def __init__(self, target_no_data_value: int):
        self.target_no_data_value = target_no_data_value
        self.running_sum = 0.0
        self.count = 0

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            nan_target_mask = y == self.target_no_data_value

            valid_mask = ~nan_target_mask

            y_hat = y_hat[valid_mask]
            y = y[valid_mask]

            if y.numel() == 0:
                logger.warning(
                    "No valid target values found, skipping metric computation!"
                )
                return float("nan")

            result = self.compute(y_hat, y).item()
            self.running_sum += result
            self.count += 1
            return result

    @abstractmethod
    def compute(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pass

    def aggregate(self) -> float:
        if self.count == 0:
            return float("nan")
        aggregated_result = self.running_sum / self.count
        self.running_sum = 0.0
        self.count = 0
        return aggregated_result
