import torch.nn as nn
from losses.dice import BinarySoftDiceLoss


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "ce_loss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "dice_loss":
        return BinarySoftDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'")
