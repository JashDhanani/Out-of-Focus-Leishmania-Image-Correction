import torch.nn as nn


class PatchDiscriminator(nn.Module):
    """70x70 Markovian (PatchGAN) discriminator.

    Purely convolutional; the output is an n x n map of logits, one per
    receptive-field patch, matching the "n x n matrix" description in the
    paper. Real/fake probability is the mean over that map.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        channels = base_channels
        for i in range(1, n_layers):
            out_channels = min(channels * 2, 512)
            layers += [
                nn.Conv2d(channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = out_channels

        out_channels = min(channels * 2, 512)
        layers += [
            nn.Conv2d(channels, out_channels, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = out_channels

        layers += [nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
