import torch.nn as nn
from losses.dice import BinaryDiceLoss


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "dice_loss":
        return BinaryDiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'")
