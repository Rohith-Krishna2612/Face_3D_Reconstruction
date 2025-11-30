"""
Discriminator networks for GAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBlock(nn.Module):
    """Convolutional block with normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4, 
                 stride: int = 2, padding: int = 1, use_norm: bool = True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels) if use_norm else nn.Identity()
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(x)))


class Discriminator(nn.Module):
    """PatchGAN Discriminator for face restoration."""
    
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(ConvBlock(in_channels, ndf, use_norm=False))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult))
        
        # Penultimate layer
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers.append(ConvBlock(ndf * nf_mult_prev, ndf * nf_mult, stride=1))
        
        # Final layer
        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """Multi-scale discriminator for better training stability."""
    
    def __init__(self, in_channels: int = 3, ndf: int = 64, n_layers: int = 3, num_scales: int = 3):
        super().__init__()
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(num_scales):
            self.discriminators.append(Discriminator(in_channels, ndf, n_layers))
            
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        results = []
        
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            results.append(disc(x))
            
        return results


def create_discriminator(config: dict) -> Discriminator:
    """Factory function to create discriminator."""
    return Discriminator(in_channels=3, ndf=64, n_layers=3)


if __name__ == "__main__":
    # Test discriminator
    disc = Discriminator()
    x = torch.randn(1, 3, 256, 256)
    
    with torch.no_grad():
        out = disc(x)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test multiscale discriminator
    multi_disc = MultiscaleDiscriminator()
    results = multi_disc(x)
    
    print(f"Multiscale results: {[r.shape for r in results]}")