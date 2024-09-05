import torch.nn as nn
from losses.dice import DiceLoss
from losses.ce_dice import BinaryCeSoftDiceLoss


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "ce_loss":
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == "dice_loss":
        return DiceLoss(**kwargs)
    elif loss_name == "ce_dice_loss":
        return BinaryCeSoftDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'")
