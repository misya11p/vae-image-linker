import torch
from torch import nn


class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        skip = self.skip_conv(x)
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.conv2(z)
        z = self.bn2(z) + skip
        z = self.act(z)
        return z


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x):
        skip = self.skip_conv(self.upsample(x))
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.act(z)
        z = self.upsample(z)
        z = self.conv2(z)
        z = self.bn2(z) + skip
        z = self.act(z)
        return z


class Encoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.net = nn.Sequential(
            ResDownBlock(3, 32, 4, 2, 1), # 48
            ResDownBlock(32, 64, 4, 2, 1), # 24
            ResDownBlock(64, 128, 4, 2, 1), # 12
            ResDownBlock(128, 256, 4, 2, 1), # 6
            ResDownBlock(256, 256, 4, 2, 1), # 3
        )
        self.conv_mean = nn.Conv2d(256, z_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, z_dim, kernel_size=1)

    def forward(self, x):
        y = self.net(x)
        mean = self.conv_mean(y)
        logvar = self.conv_logvar(y)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            ResUpBlock(z_dim, 256), # 6
            ResUpBlock(256, 128), # 12
            ResUpBlock(128, 64), # 24
            ResUpBlock(64, 32), # 48
            ResUpBlock(32, 3), # 96
            nn.Sigmoid(),
        )

    def forward(self, z):
        y = self.net(z)
        return y


class VAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        y = self.decoder(z)
        return y, mean, logvar

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
