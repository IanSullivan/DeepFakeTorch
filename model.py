from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, inputs, size=512):
        # print(input.view(input.size(0), size, 1, 1).shape, 'fffff')
        # return torch.reshape(input, (input.size(0), 32, 4, 4))
        return inputs.view(inputs.size(0), 128, 4, 4)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Flatten(),
            # nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


class AutoEncoder(nn.Module):

    def __init__(self, image_channels=3, h_dim=2048, z_dim=128):
        super(AutoEncoder, self).__init__()
        self.device = 'cuda'
        self.encoder = nn.Sequential(
            Downscale(image_channels, 64),
            Downscale(64, 128),
            # Downscale(128, 128),
            # ResBlock(128),
            Downscale(128, 256),
            # ResBlock(256),
            Downscale(256, 256),
            Downscale(256, 512),
            # ResBlock(512),
            Downscale(512, 512),
            Flatten(),
        )
        # ([32, 2304])

        self.inter_layer = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Linear(z_dim, z_dim),
            nn.Linear(z_dim, h_dim),
        )

        self.decoder = nn.Sequential(
            UnFlatten(),
            # Upscale(128, 128, kernel_size=4),
            Upscale(128, 256, kernel_size=4),
            # ResBlock(256),
            # ResBlock(128),
            # ResBlock(128),
            Upscale(256, 256, kernel_size=4),
            Upscale(256, 128, kernel_size=4),
            # ResBlock(128),
            Upscale(128, 64, kernel_size=4),
            ResBlock(64),
            Upscale(64, 32, kernel_size=4),
            Upscale(32, 32, kernel_size=4),
            nn.Conv2d(32, image_channels, kernel_size=1, stride=2),
            nn.Sigmoid(),
        )

        self.decoder_b = nn.Sequential(
            UnFlatten(),
            # Upscale(128, 128, kernel_size=4),
            Upscale(128, 256, kernel_size=4),
            # ResBlock(256),
            # ResBlock(128),
            # ResBlock(128),
            Upscale(256, 256, kernel_size=4),
            Upscale(256, 128, kernel_size=4),
            # ResBlock(128),
            Upscale(128, 64, kernel_size=4),
            ResBlock(64),
            Upscale(64, 32, kernel_size=4),
            Upscale(32, 32, kernel_size=4),
            nn.Conv2d(32, image_channels, kernel_size=1, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, x, version='a'):
        z = self.encoder(x)
        z = self.inter_layer(z)
        if version == 'a':
            z = self.decoder(z)
        else:
            z = self.decoder_b(z)
        return z


class ResBlock(nn.Module):
    def __init__(self, n_ch) -> None:
        super().__init__()

        self.resblock_model = nn.Sequential(
            nn.Conv2d(n_ch, n_ch, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(n_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(n_ch, n_ch, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(n_ch)
        )

    def forward(self, inputs):
        return self.resblock_model(inputs) + inputs


class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(self.out_ch)
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout2d()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout2d()

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.drop(x)
        return x
