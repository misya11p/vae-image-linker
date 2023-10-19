import torch
from torch import nn
import torch.nn.functional as F


class SimpleVAE(nn.Module):
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




# ----------------------------------------------------------------------
# Reference: CNN-VAE/RES_VAE_64_old.py at master · LukeDitria/CNN-VAE

#Residual down sampling block for the encoder
#Average pooling is used to perform the downsampling
class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_down, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale,scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


#Residual up sampling block for the decoder
#Nearest neighbour is used to perform the upsampling
class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale = 2):
        super(Res_up, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out//2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out//2)
        self.conv2 = nn.Conv2d(channel_out//2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.UpNN = nn.Upsample(scale_factor = scale,mode = "nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x

#Encoder block
#Built for a 64x64x3 image
class Encoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)
        self.conv2 = Res_down(ch, 2*ch)
        self.conv3 = Res_down(2*ch, 4*ch)
        self.conv4 = Res_down(4*ch, 8*ch)
        self.conv_mu = nn.Conv2d(8*ch, z, 4, 1)
        self.conv_logvar = nn.Conv2d(8*ch, z, 4, 1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # if self.training:
        if 1:
            mu = self.conv_mu(x)
            logvar = self.conv_logvar(x)
            x = self.sample(mu, logvar)
        else:
            mu = self.conv_mu(x)
            mu = x
            logvar = None
        return x, mu, logvar

#Decoder block
#Built to be a mirror of the encoder block
class Decoder(nn.Module):
    def __init__(self, channels, ch = 64, z = 512):
        super(Decoder, self).__init__()
        self.conv1 = Res_up(z, ch*8, scale = 4)
        self.conv2 = Res_up(ch*8, ch*4)
        self.conv3 = Res_up(ch*4, ch*2)
        self.conv4 = Res_up(ch*2, ch)
        # self.conv5 = Res_up(ch, ch//2)
        # self.conv6 = nn.Conv2d(ch//2, channels, 3, 1, 1)
        self.conv6 = nn.Conv2d(ch, channels, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = self.conv6(x)
        x = self.sigmoid(x)

        return x

#VAE network, uses the above encoder and decoder blocks
class VAE(nn.Module):
    def __init__(self, channel_in, z = 512):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image
        z = the number of channels of the latent representation"""

        self.encoder = Encoder(channel_in, z = z)
        self.decoder = Decoder(channel_in, z = z)

    def forward(self, x):
        encoding, mu, logvar = self.encoder(x)
        recon = self.decoder(encoding)
        return recon, mu, logvar
