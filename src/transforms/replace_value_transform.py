import torch.nn as nn
from typing import Optional
from torchvision import tv_tensors


class ReplaceValueTransform(nn.Module):
    def __init__(self, value_to_replace: float, replace_value: float):
        super().__init__()
        self.value_to_replace = value_to_replace
        self.replace_value = replace_value

    def forward(self, img: tv_tensors.Image, mask: Optional[tv_tensors.Mask]=None):
        img[img == self.value_to_replace] = self.replace_value
        
        if mask is None:
            return img
        
        return img, mask