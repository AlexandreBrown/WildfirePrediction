import torch.nn as nn
from typing import Optional
from torchvision import tv_tensors
import torch


class ReplaceNanValueTransform(nn.Module):
    def __init__(self, replace_value: float):
        super().__init__()
        self.replace_value = replace_value

    def forward(self, img: tv_tensors.Image, mask: Optional[tv_tensors.Mask] = None):
        img = torch.nan_to_num(img, nan=self.replace_value)
        if mask is None:
            return img
        return img, mask
