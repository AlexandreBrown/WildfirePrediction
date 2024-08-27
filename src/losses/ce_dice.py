import torch
import torch.nn as nn
from losses.dice import BinarySoftDiceLoss


class BinaryCeSoftDiceLoss(nn.Module):
    def __init__(
        self, ce_weight: float, dice_weight: float, ce_params: dict, dice_params: dict
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.BCEWithLogitsLoss(**ce_params)
        self.dice_loss = BinarySoftDiceLoss(**dice_params)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(predictions, target)
        dice_loss = self.dice_loss(predictions, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
