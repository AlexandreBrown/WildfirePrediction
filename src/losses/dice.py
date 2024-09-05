import torch
import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses


class DiceLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = smp_losses.DiceLoss(mode="binary", **kwargs)

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(predictions.unsqueeze(1), target)
