import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, image_size: int, device: str):
        super().__init__()
        self.image_size = image_size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            self._linear_block(image_size**2, 512),
            self._linear_block(512, 128),
            nn.Linear(128, 64*2),
        )
        self.decoder = nn.Sequential(
            self._linear_block(64, 128),
            self._linear_block(128, 512),
            nn.Linear(512, image_size**2),
            nn.Sigmoid()
        )
        self.device = device
        self.to(device)

    @staticmethod
    def _linear_block(input_size, output_size):
        net = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
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
        y = self.decoder(z).reshape(-1, self.image_size, self.image_size)
        return y

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path, map_location=self.device))
