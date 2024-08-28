import torch
from torchmetrics.functional.classification import binary_average_precision
from metrics.metric import Metric


class PrecisionRecallAucMetric(Metric):
    def __init__(self, target_no_data_value: int):
        super().__init__(target_no_data_value)
        self.name = "pr_auc"

    def compute(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return binary_average_precision(y_hat, y)
