"""
CodeFormer model implementation for face restoration.
Based on the original CodeFormer paper: https://arxiv.org/abs/2206.11253
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class VectorQuantizer(nn.Module):
    """Vector Quantization module for the codebook."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))
        
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Convert quantized from BHWC -> BCHW
        return quantized.permute(0, 3, 1, 2).contiguous(), loss, perplexity


class ResidualBlock(nn.Module):
    """Residual block with group normalization."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(32, out_channels)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        
        return F.relu(out + residual)


class AttentionBlock(nn.Module):
    """Self-attention block for CodeFormer."""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Reshape to sequence
        x_seq = x.view(B, C, H*W).transpose(1, 2)  # B, HW, C
        
        # Self-attention
        x_norm = self.norm(x_seq)
        qkv = self.qkv(x_norm).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        out = self.proj(out)
        
        # Residual connection
        out = x_seq + out
        
        # Reshape back
        out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


class VQGANEncoder(nn.Module):
    """VQGAN Encoder for CodeFormer."""
    
    def __init__(self, in_channels: int = 3, hidden_dim: int = 512, codebook_dim: int = 256):
        super().__init__()
        
        self.conv_in = nn.Conv2d(in_channels, 128, 3, padding=1)
        
        # Downsampling blocks
        self.down1 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 512 -> 256
        )
        
        self.down2 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 256 -> 128
        )
        
        self.down3 = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Conv2d(512, hidden_dim, 4, stride=2, padding=1)  # 128 -> 64
        )
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            AttentionBlock(hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        
        # Final projection to codebook dimension
        self.conv_out = nn.Conv2d(hidden_dim, codebook_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.middle(x)
        x = self.conv_out(x)
        return x


class VQGANDecoder(nn.Module):
    """VQGAN Decoder for CodeFormer."""
    
    def __init__(self, out_channels: int = 3, hidden_dim: int = 512, codebook_dim: int = 256):
        super().__init__()
        
        self.conv_in = nn.Conv2d(codebook_dim, hidden_dim, 1)
        
        # Middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            AttentionBlock(hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )
        
        # Upsampling blocks
        self.up1 = nn.Sequential(
            ResidualBlock(hidden_dim, 512),
            ResidualBlock(512, 512),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 64 -> 128
        )
        
        self.up2 = nn.Sequential(
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 128 -> 256
        )
        
        self.up3 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 256 -> 512
        )
        
        self.conv_out = nn.Conv2d(64, out_channels, 3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.middle(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = torch.tanh(self.conv_out(x))
        return x


class CodeFormer(nn.Module):
    """Complete CodeFormer model."""
    
    def __init__(
        self, 
        dim: int = 512,
        codebook_size: int = 1024, 
        codebook_dim: int = 256,
        n_head: int = 8,
        n_layers: int = 9,
        connect_list: list = None
    ):
        super().__init__()
        
        if connect_list is None:
            connect_list = ['32', '64', '128', '256']
            
        self.connect_list = connect_list
        self.n_layers = n_layers
        
        # Encoder
        self.encoder = VQGANEncoder(3, dim, codebook_dim)
        
        # Vector Quantization
        self.quantize = VectorQuantizer(codebook_size, codebook_dim)
        
        # Decoder
        self.decoder = VQGANDecoder(3, dim, codebook_dim)
        
        # Transformer layers for code prediction
        self.code_pred_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=codebook_dim,
                nhead=n_head,
                dim_feedforward=codebook_dim * 4,
                dropout=0.0,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Feature extraction for multi-scale fusion
        self.feat_extract_layers = nn.ModuleDict()
        for size in connect_list:
            self.feat_extract_layers[size] = nn.Conv2d(codebook_dim, codebook_dim, 1)
            
    def forward(self, x: torch.Tensor, w: float = 0.0) -> Tuple[torch.Tensor, dict]:
        """Forward pass with controllable restoration strength w."""
        
        # Encode
        feat = self.encoder(x)
        
        # Quantize
        quant_feat, vq_loss, perplexity = self.quantize(feat)
        
        if w == 0:  # No code prediction, direct decoding
            out = self.decoder(quant_feat)
        else:
            # Code prediction with transformer
            B, C, H, W = feat.shape
            
            # Flatten features for transformer
            feat_flat = feat.view(B, C, H*W).transpose(1, 2)  # B, HW, C
            
            # Apply transformer layers
            pred_feat = feat_flat
            for layer in self.code_pred_layers:
                pred_feat = layer(pred_feat, feat_flat)
            
            # Reshape back
            pred_feat = pred_feat.transpose(1, 2).view(B, C, H, W)
            
            # Mix original and predicted features based on w
            mixed_feat = (1 - w) * quant_feat + w * pred_feat
            
            # Decode
            out = self.decoder(mixed_feat)
        
        info = {
            'vq_loss': vq_loss,
            'perplexity': perplexity
        }
        
        return out, info


def create_codeformer_model(config: dict) -> CodeFormer:
    """Factory function to create CodeFormer model."""
    model_config = config['model']
    
    return CodeFormer(
        dim=512,
        codebook_size=model_config['codebook_size'],
        codebook_dim=model_config['code_dim'],
        n_head=8,
        n_layers=9
    )


if __name__ == "__main__":
    # Test model
    model = CodeFormer()
    x = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        out, info = model(x, w=0.5)
        
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"VQ loss: {info['vq_loss']:.4f}")
    print(f"Perplexity: {info['perplexity']:.4f}")