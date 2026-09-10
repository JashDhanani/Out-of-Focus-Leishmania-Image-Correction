import torch
import torch.nn as nn

from .models.vgg import VGGContentExtractor


class GANLoss(nn.Module):
    """Vanilla (non-saturating) adversarial loss, Eq. (5) in the paper.

    The discriminator outputs an n x n patch map of logits; BCEWithLogits
    reduced with 'mean' is equivalent to averaging log-probabilities over the
    patch grid, matching "the discriminate probability is the average value
    in the n x n matrix".
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def __call__(self, prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
        target = torch.ones_like(prediction) if is_real else torch.zeros_like(prediction)
        return self.loss(prediction, target)


class ContentLoss(nn.Module):
    """Perceptual (VGG feature-space) content loss, Eq. (6)-(8)."""

    def __init__(self, layer_index: int | None = None):
        super().__init__()
        kwargs = {} if layer_index is None else {"layer_index": layer_index}
        self.vgg = VGGContentExtractor(**kwargs)
        self.criterion = nn.MSELoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        feat_gen = self.vgg(generated)
        with torch.no_grad():
            feat_target = self.vgg(target)
        return self.criterion(feat_gen, feat_target)


class CycleLoss(nn.Module):
    """Cycle-consistency L1 loss, Eq. (9)."""

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        return self.criterion(reconstructed, original)
