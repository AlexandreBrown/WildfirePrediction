import torch
import torch.nn as nn
from loguru import logger


class NanAwareLoss(nn.Module):
    def __init__(self, loss: nn.Module, target_no_data_value: int):
        super().__init__()
        self.loss = loss
        self.target_no_data_value = target_no_data_value

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        nan_target_mask = target == self.target_no_data_value

        valid_mask = ~nan_target_mask

        predictions = predictions[valid_mask]
        target = target[valid_mask]

        if target.numel() == 0:
            logger.warning("No valid target values found, returning 0.0 loss!")
            return torch.tensor(
                0.0, device=predictions.device, requires_grad=predictions.requires_grad
            )

        return self.loss(predictions, target)
