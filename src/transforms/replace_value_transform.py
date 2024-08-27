import torch.nn as nn
from torchvision import tv_tensors


class ReplaceNoDataValueTransform(nn.Module):
    def __init__(self, nodata_value: float, replace_value: float):
        super().__init__()
        self.nodata_value = nodata_value
        self.replace_value = replace_value

    def forward(self, img: tv_tensors.Image, mask: tv_tensors.Mask) -> tuple:
        mask = img == self.nodata_value
        img[mask] = self.replace_value
        return img, mask
