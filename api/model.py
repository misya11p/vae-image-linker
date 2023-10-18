import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, image_size: int, device: str):
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(
            self._conv_block(3, 32, 7, 2, 3), # 128 -> 32
            self._conv_block(32, 64, 5, 2, 2), # 32 -> 8
            self._conv_block(64, 128, 3, 2, 1), # 8 -> 2
            nn.Flatten(),
            nn.Linear(128*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64*2),
        )
        self.decoder = nn.Sequential(
            self._conv_t_block(64, 32, 4, 1, 0), # 1 -> 4
            self._conv_t_block(32, 16, 6, 4, 1), # 4 -> 16
            self._conv_t_block(16, 8, 6, 4, 1), # 16 -> 64 
            nn.ConvTranspose2d(8, 3, 4, 2, 1), # 64 -> 128
            nn.Sigmoid()
        )
        self.device = device
        self.to(device)

    @staticmethod
    def _conv_block(in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return net

    @staticmethod
    def _conv_t_block(in_channels, out_channels, kernel_size, stride, padding):
        net = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        return net

    def forward(self, x):
        z, mean, log_var = self.encode(x)
        y = self.decode(z)
        return y, mean, log_var, z

    def representation(self, mean, log_var):
        norm = torch.randn_like(mean)
        z = mean + torch.exp(log_var/2)*norm
        return z

    def encode(self, x):
        mean, log_var = self.encoder(x).chunk(2, dim=1)
        z = self.representation(mean, log_var).to(self.device)
        return z, mean, log_var

    def decode(self, z):
        z = z.reshape(-1, 64, 1, 1)
        y = self.decoder(z)
        return y

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
