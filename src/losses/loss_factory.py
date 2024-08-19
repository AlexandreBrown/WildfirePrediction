import torch.nn as nn


def create_loss(loss_name: str):
    if loss_name == "bce_logits":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss name: '{loss_name}'")
