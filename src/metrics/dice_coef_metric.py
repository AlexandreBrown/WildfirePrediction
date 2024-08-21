import torch
import torchmetrics.segmentation


class DiceCoefMetric:
    def __init__(self, nb_classes: int):
        self.name = "dice_coef"
        self.nb_classes = nb_classes
        self._metric = torchmetrics.segmentation.GeneralizedDiceScore(
            num_classes=nb_classes
        )

    def __call__(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_hat_binary = (y_hat > 0).long()
        return self._metric(y_hat_binary, y)

    def compute(self):
        return self._metric.compute()
