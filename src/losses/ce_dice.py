import torch
import torch.nn as nn
from losses.dice import DiceLoss


class BinaryCeSoftDiceLoss(nn.Module):
    def __init__(
        self,
        device,
        ce_weight: float,
        dice_weight: float,
        ce_params: dict,
        dice_params: dict,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        if "pos_weight" in ce_params:
            ce_params["pos_weight"] = torch.tensor(list(ce_params["pos_weight"])).to(
                device
            )
        self.ce_loss = nn.BCEWithLogitsLoss(**ce_params)
        self.dice_loss = DiceLoss(**dice_params)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(predictions, target)
        dice_loss = self.dice_loss(predictions, target)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss
