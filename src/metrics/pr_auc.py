import torch
from torchmetrics.functional.classification import binary_average_precision


class PrecisionRecallAuc:
    def __init__(self):
        self.name = "pr_auc"
        self.running_sum = 0.0
        self.count = 0

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        pr_auc = binary_average_precision(y_hat, y)
        self.running_sum += pr_auc.item()
        self.count += 1
        return pr_auc.item()

    def compute(self) -> float:
        return self.running_sum / self.count
