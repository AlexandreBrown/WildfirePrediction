import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from torchvision import tv_tensors


class ConditionalNormalize(nn.Module):
    def __init__(self, mean, std, nodata_value):
        super().__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.nodata_value = nodata_value

    def forward(self, img: tv_tensors.Image, mask: tv_tensors.Mask) -> tuple:

        normalized_img = F.normalize(img, self.mean, self.std)

        nodata_mask = img == self.nodata_value

        normalized_img[nodata_mask] = img[nodata_mask]

        return normalized_img, mask
