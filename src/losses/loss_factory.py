import torch
import torch.nn as nn
from losses.dice import DiceLoss
from losses.ce_dice import BinaryCeSoftDiceLoss


def create_loss(device, loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "ce_loss":
        if "pos_weight" in kwargs:
            kwargs["pos_weight"] = torch.tensor(list(kwargs["pos_weight"])).to(device)
        return nn.BCEWithLogitsLoss(**kwargs)
    elif loss_name == "dice_loss":
        return DiceLoss(**kwargs)
    elif loss_name == "ce_dice_loss":
        return BinaryCeSoftDiceLoss(device, **kwargs)
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'")
