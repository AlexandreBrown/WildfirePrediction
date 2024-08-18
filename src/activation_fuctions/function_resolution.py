import torch


def get_activation(name: str):
    if name == "relu":
        return torch.relu
    elif name == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError(f"Activation function {name} not supported")
