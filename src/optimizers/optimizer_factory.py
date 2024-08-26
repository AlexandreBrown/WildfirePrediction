import torch
import torch.nn as nn


def create_optimizer(model: nn.Module, optimizer_config: dict):
    optimizer_name = optimizer_config["name"]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config["params"])
    else:
        raise ValueError(f"Unknown optimizer name: '{optimizer_name}'")

    return optimizer
