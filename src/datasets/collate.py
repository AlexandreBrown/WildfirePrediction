import torch.nn as nn
from datasets.wildfire_data import WildfireData


class CollateDataAugs(nn.Module):
    def __init__(self, data_augs=None, device=None):
        super().__init__()
        self.data_augs = data_augs
        self.device = device

    def __call__(self, x: WildfireData):
        if self.device is not None:
            out = x.to(self.device)
        else:
            out = x
        if self.data_augs is not None:
            out.images, out.masks = self.data_augs(out.images, out.masks)
        return out
