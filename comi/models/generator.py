import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """A single residual block with reflection padding and instance norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Style-transfer generator (Johnson et al.) used by CycleGAN.

    Encoder: 7x7 conv + two stride-2 conv blocks (downsampling).
    Bottleneck: `n_blocks` residual blocks (9 by default, as in the paper).
    Decoder: two stride-2 transposed conv blocks (upsampling) + 7x7 conv.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, n_blocks: int = 9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        # two stride-convolution (downsampling) blocks
        channels = base_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        # residual blocks
        for _ in range(n_blocks):
            layers += [ResnetBlock(channels)]

        # two transposed-convolution (upsampling) blocks
        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(channels, channels // 2, kernel_size=3, stride=2,
                                    padding=1, output_padding=1),
                nn.InstanceNorm2d(channels // 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
