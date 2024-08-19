import torch
import torch.nn as nn


def create_optimizer(model: nn.Module, optimizer_name: str, lr: float):
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer name: '{optimizer_name}'")

    return optimizer
