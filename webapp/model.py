"""Standalone copy of comi.models.generator.ResnetGenerator for deployment.

Kept dependency-free (torch only) so the hosted demo doesn't need the full
training stack (VGG perceptual loss, scipy/scikit-image metrics, etc.) --
inference only needs the generator architecture and its exported weights.
"""
import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
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
    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_channels: int = 64, n_blocks: int = 9):
        super().__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels, affine=True),
            nn.ReLU(inplace=True),
        ]

        channels = base_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2, affine=True),
                nn.ReLU(inplace=True),
            ]
            channels *= 2

        for _ in range(n_blocks):
            layers += [ResnetBlock(channels)]

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


def load_generator(weights_path: str, base_channels: int = 64, n_blocks: int = 9) -> ResnetGenerator:
    model = ResnetGenerator(base_channels=base_channels, n_blocks=n_blocks)
    state = torch.load(weights_path, map_location="cpu")
    state = {k: v.float() for k, v in state.items()}  # fp16-exported weights -> fp32 for CPU inference
    model.load_state_dict(state)
    model.eval()
    return model
