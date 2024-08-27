import torch


class LossMetric:
    def __init__(self, loss, name: str):
        self.name = name
        self.running_sum = 0.0
        self.count = 0
        self.loss = loss

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        with torch.no_grad():
            loss_result = self.loss(y_hat, y.float())
            self.running_sum += loss_result.item()
            self.count += 1
            return loss_result.item()

    def compute(self) -> float:
        return self.running_sum / self.count
