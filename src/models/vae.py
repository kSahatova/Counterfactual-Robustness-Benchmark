import torch
from torch import nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple


class Encoder(nn.Module):
    def __init__(self, input_channels, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 32, kernel_size=4, stride=2, padding=1
        )  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=4, stride=2, padding=1
        )  # 14x14 -> 7x7
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=2, padding=1
        )  # 7x7 -> 4x4
        self.fc_mu = nn.Linear(128 * 4 * 4, z_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, z_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, z_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(z_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # 4x4 -> 8x8
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=4, stride=2, padding=1
        )  # 8x8 -> 16x16
        self.deconv3 = nn.ConvTranspose2d(
            32, output_channels, kernel_size=4, stride=2, padding=1
        )  # 16x16 -> 32x32
        self.output_layer = nn.Conv2d(
            output_channels, output_channels, kernel_size=5, stride=1, padding=0
        )  # 32x32 -> 28x28

    def forward(self, z):
        h = F.relu(self.fc(z))
        h = h.view(-1, 128, 4, 4)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        x_recon = torch.sigmoid(self.output_layer(h))
        return x_recon


class VAE(nn.Module):
    def __init__(self, input_channels, z_dim, beta):
        super(VAE, self).__init__()
        self.beta = beta
        self.encoder = Encoder(input_channels, z_dim)
        self.decoder = Decoder(z_dim, input_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar


@dataclass
class VAEConfig:
    """Configuration for VAE architecture"""
    input_channels: int = 3
    hidden_dims: List[int] = None
    latent_dim: int = 128
    input_size: Tuple[int, int] = (224, 224)
    beta: int = 1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            # Default architecture for 224x224 images
            # Each conv layer will divide spatial dimensions by 2
            # 224 -> 112 -> 56 -> 28 -> 14 -> 7
            self.hidden_dims = [32, 64, 128, 256, 512]


class BetaVAE(nn.Module):
    """
    Another implementation of beta-VAE for more flexible adjustment of the number of hidden layers 
    """
    def __init__(self, config: VAEConfig):
        super().__init__()
        self.config = config
        self.beta = config.beta
        self.latent_dim = config.latent_dim
        
        # Calculate the size of the flattened feature map
        self.feature_size = (
            config.input_size[0] // (2 ** len(config.hidden_dims)),
            config.input_size[1] // (2 ** len(config.hidden_dims))
        )
        self.flattened_dim = config.hidden_dims[-1] * self.feature_size[0] * self.feature_size[1]
        
        # Build Encoder
        modules = []
        in_channels = config.input_channels
        
        for h_dim in config.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=4, stride=2, padding=1),
                    #nn.BatchNorm2d(h_dim),
                    nn.ReLU()
                )
            )
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.flattened_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.flattened_dim, self.latent_dim)
        
        # Build Decoder
        modules = []
        
        self.decoder_input = nn.Linear(self.latent_dim, self.flattened_dim)
        
        hidden_dims_reversed = config.hidden_dims[::-1]
        
        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims_reversed[i],
                        hidden_dims_reversed[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    #nn.BatchNorm2d(hidden_dims_reversed[i + 1]),
                    nn.ReLU()
                )
            )
        
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                hidden_dims_reversed[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            #nn.BatchNorm2d(hidden_dims_reversed[-1]),
            nn.ReLU(),
            nn.Conv2d(hidden_dims_reversed[-1], config.input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.config.hidden_dims[-1], self.feature_size[0], self.feature_size[1])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var
    
