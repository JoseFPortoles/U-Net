import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)  # Nueva capa en la ruta de contracción

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=1)
        self.upconv1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=1)
        self.upconv2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=1)
        self.upconv3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=1)
        self.upconv4 = DoubleConv(128, 64)

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Ruta de contracción
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)  # Nueva capa agregada

        # Ruta de expansión
        x6 = self.up1(x5)
        x6 = torch.cat([x6, x4[:, :, :x6.size(2), :x6.size(3)]], dim=1)
        x6 = self.upconv1(x6)

        x7 = self.up2(x6)
        x7 = torch.cat([x7, x3[:, :, :x7.size(2), :x7.size(3)]], dim=1)
        x7 = self.upconv2(x7)

        x8 = self.up3(x7)
        x8 = torch.cat([x8, x2[:, :, :x8.size(2), :x8.size(3)]], dim=1)
        x8 = self.upconv3(x8)

        x9 = self.up4(x8)
        x9 = torch.cat([x9, x1[:, :, :x9.size(2), :x9.size(3)]], dim=1)
        x9 = self.upconv4(x9)

        # Capa final
        output = self.outconv(x9)
        return output, [x1, x2, x3, x4, x5], [x6, x7, x8, x9]