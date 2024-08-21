import torch
import torchmetrics.classification


class PrecisionRecallAuc:
    def __init__(self):
        self.name = "pr_auc"
        self._metric = torchmetrics.classification.BinaryAveragePrecision()

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor):
        return self._metric(y_hat, y)

    def compute(self):
        return self._metric.compute()
