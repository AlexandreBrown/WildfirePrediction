import torch
from metrics.metric import Metric


class LossMetric(Metric):
    def __init__(self, target_no_data_value: int, loss: torch.nn.Module, name: str):
        super().__init__(target_no_data_value)
        self.loss = loss
        self.name = name

    def compute(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(y_hat, y.float())
