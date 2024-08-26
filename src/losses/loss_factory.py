import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    if loss_name == "ce_loss":
        return nn.BCEWithLogitsLoss()
    elif loss_name == "dice_loss":
        return DiceLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'")
