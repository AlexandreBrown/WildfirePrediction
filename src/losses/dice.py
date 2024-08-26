import torch
import torch.nn.functional as F


class BinarySoftDiceLoss(torch.nn.Module):
    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.compute_dice_loss(predictions, target)

    def compute_dice_loss(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return 1.0 - self.compute_soft_dice_score(predictions, target)

    def compute_soft_dice_score(
        self, predictions: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        assert predictions.shape == target.shape

        if self.from_logits:
            predictions = F.logsigmoid(predictions).exp()

        predictions = predictions.reshape(predictions.shape[0], 1, -1)
        target = target.reshape(target.shape[0], 1, -1)

        intersection = torch.sum(predictions * target, dim=-1)
        cardinality = torch.sum(predictions + target, dim=-1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return torch.mean(dice_score)
