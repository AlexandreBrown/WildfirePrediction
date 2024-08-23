import torch.nn as nn
from datasets.wildfire_data import WildfireData


class Collate(nn.Module):
    def __init__(self, transform=None, device=None):
        super().__init__()
        self.transform = transform
        self.device = device

    def __call__(self, x: WildfireData):
        if self.device is not None:
            if self.device.type == "cuda":
                out = x.pin_memory()
            else:
                out = x
            out = out.to(self.device)
        else:
            out = x
        if self.transform is not None:
            out.images, out.masks = self.transform(out.images, out.masks)
        return out
