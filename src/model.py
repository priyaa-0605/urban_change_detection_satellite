import torch
import torch.nn as nn

# ----- Basic Convolution Block -----
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ----- U-Net Model -----
class UNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=1):
        super().__init__()

        # Encoder
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, before, after):
        # Concatenate before & after images
        x = torch.cat([before, after], dim=1)

        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        bn = self.bottleneck(p3)

        # Decoder
        u3 = self.up3(bn)
        c3 = self.conv3(torch.cat([u3, d3], dim=1))

        u2 = self.up2(c3)
        c2 = self.conv2(torch.cat([u2, d2], dim=1))

        u1 = self.up1(c2)
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        return self.out(c1)
