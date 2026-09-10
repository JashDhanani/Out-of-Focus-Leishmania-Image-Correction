import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# index of the layer output to use for the content/perceptual loss:
# 26 -> output of relu4_4 in torchvision's vgg19.features, i.e. the
# activations "after the j-th convolution, before the i-th maxpooling layer"
# referenced in Eq. 6/7 of the paper.
DEFAULT_LAYER_INDEX = 26


class VGGContentExtractor(nn.Module):
    """Frozen VGG-19 feature extractor used for the perceptual content loss."""

    def __init__(self, layer_index: int = DEFAULT_LAYER_INDEX):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*list(vgg.children())[: layer_index + 1]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # inputs are assumed to be in [-1, 1] (tanh output range); rescale to
        # [0, 1] then normalize with ImageNet statistics before feeding VGG.
        x = (x + 1.0) / 2.0
        x = (x - self.mean) / self.std
        return self.slice(x)
