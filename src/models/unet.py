from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """([BN] => convolution  => ReLU) * 2"""

    def __init__(self, channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResUnit(nn.Module):
    """Residual unit"""

    def __init__(self, channels):
        super().__init__()
        self.doubleconv = DoubleConv(channels)

    def forward(self, x):
        return self.doubleconv(x) + x


class Down(nn.Module):
    """3x Residual unit, then strided conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.res = nn.Sequential(
            ResUnit(in_channels),
            ResUnit(in_channels),
            ResUnit(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
        )

    def forward(self, x):
        return self.res(x)


class UpscoreUnit(nn.Module):
    """Upscore unit as in the paper"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d = (in_channels, out_channels)
        self.convbn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # The paper is lying-
        # https://github.com/vanya2v/Multi-modal-learning/blob/628acee4e275db16733b13e4b1ae766132030b28/dltk/models/segmentation/fcn.py#L57
        self.conv_filtercorrection = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x1, x2):
        x1 = self.convbn(x1)
        x2 = self.conv_filtercorrection(x2)

        return x1 + x2


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unit = UpscoreUnit(in_channels, out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

    def forward(self, x1, x2):
        x2 = self.upsample(x2)
        return self.unit(x1, x2)


class Unet(nn.Module):
    def __init__(self, n_channels=1, n_classes=8):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Conv2d(n_channels, 16, 3, padding=1)
        self.down1 = Down(16, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 16)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        return self.outc(x)
