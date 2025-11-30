"""
Loss functions for CodeFormer training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    
    def __init__(self, layer_weights: dict = None):
        super().__init__()
        
        if layer_weights is None:
            layer_weights = {
                'conv1_2': 0.1,
                'conv2_2': 0.1,
                'conv3_4': 1.0,
                'conv4_4': 1.0,
                'conv5_4': 1.0
            }
        
        self.layer_weights = layer_weights
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        self.vgg_layers = nn.ModuleDict()
        
        layer_mapping = {
            'conv1_2': 3,   # after conv1_2, relu
            'conv2_2': 8,   # after conv2_2, relu
            'conv3_4': 17,  # after conv3_4, relu
            'conv4_4': 26,  # after conv4_4, relu
            'conv5_4': 35   # after conv5_4, relu
        }
        
        for name, idx in layer_mapping.items():
            if name in layer_weights:
                self.vgg_layers[name] = nn.Sequential(*list(vgg.children())[:idx+1])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
            
        # Normalization for ImageNet
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input for VGG."""
        return (x - self.mean) / self.std
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        
        # Normalize inputs
        pred_norm = self.normalize_input(pred)
        target_norm = self.normalize_input(target)
        
        loss = 0.0
        
        for layer_name, weight in self.layer_weights.items():
            pred_feat = self.vgg_layers[layer_name](pred_norm)
            target_feat = self.vgg_layers[layer_name](target_norm)
            
            loss += weight * F.mse_loss(pred_feat, target_feat)
            
        return loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training."""
    
    def __init__(self, loss_type: str = 'lsgan'):
        super().__init__()
        self.loss_type = loss_type.lower()
        
        if self.loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'wgan':
            self.criterion = None
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
            
    def forward(self, pred: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """Compute adversarial loss."""
        
        if self.loss_type == 'wgan':
            return -pred.mean() if target_is_real else pred.mean()
        
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        
        if self.loss_type == 'lsgan':
            return self.criterion(pred, target)
        else:  # vanilla
            return self.criterion(pred, target)


class CodebookLoss(nn.Module):
    """Codebook loss for vector quantization."""
    
    def __init__(self, commitment_cost: float = 0.25):
        super().__init__()
        self.commitment_cost = commitment_cost
        
    def forward(self, quantized: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """Compute codebook loss."""
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        
        return q_latent_loss + self.commitment_cost * e_latent_loss


class FacialComponentLoss(nn.Module):
    """Loss focused on facial components (eyes, nose, mouth)."""
    
    def __init__(self):
        super().__init__()
        # Could integrate with face parsing networks for component-specific loss
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute facial component loss."""
        # For now, use L1 loss. Could be enhanced with facial component masks
        return self.l1_loss(pred, target)


class TotalLoss(nn.Module):
    """Combined loss for CodeFormer training."""
    
    def __init__(self, loss_weights: dict = None):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'l1': 1.0,
                'perceptual': 1.0,
                'adversarial': 0.1,
                'codebook': 1.0,
                'component': 1.0
            }
            
        self.loss_weights = loss_weights
        
        # Initialize loss functions
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss('lsgan')
        self.codebook_loss = CodebookLoss()
        self.component_loss = FacialComponentLoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, 
                disc_pred: torch.Tensor = None, 
                vq_loss: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """Compute total loss."""
        
        losses = {}
        total_loss = 0.0
        
        # L1 loss
        if 'l1' in self.loss_weights:
            l1_loss = self.l1_loss(pred, target)
            losses['l1'] = l1_loss
            total_loss += self.loss_weights['l1'] * l1_loss
            
        # Perceptual loss
        if 'perceptual' in self.loss_weights:
            perceptual_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual_loss
            total_loss += self.loss_weights['perceptual'] * perceptual_loss
            
        # Adversarial loss
        if 'adversarial' in self.loss_weights and disc_pred is not None:
            adv_loss = self.adversarial_loss(disc_pred, True)
            losses['adversarial'] = adv_loss
            total_loss += self.loss_weights['adversarial'] * adv_loss
            
        # Codebook loss
        if 'codebook' in self.loss_weights and vq_loss is not None:
            losses['codebook'] = vq_loss
            total_loss += self.loss_weights['codebook'] * vq_loss
            
        # Component loss
        if 'component' in self.loss_weights:
            comp_loss = self.component_loss(pred, target)
            losses['component'] = comp_loss
            total_loss += self.loss_weights['component'] * comp_loss
            
        losses['total'] = total_loss
        
        return total_loss, losses


def create_loss_functions(config: dict) -> TotalLoss:
    """Factory function to create loss functions."""
    loss_weights = {
        'l1': 1.0,
        'perceptual': 1.0,
        'adversarial': 0.1,
        'codebook': 1.0,
        'component': 1.0
    }
    
    return TotalLoss(loss_weights)


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    
    # Test perceptual loss
    perceptual = PerceptualLoss().to(device)
    perc_loss = perceptual(pred, target)
    print(f"Perceptual loss: {perc_loss.item():.4f}")
    
    # Test total loss
    total_loss_fn = TotalLoss().to(device)
    total_loss, loss_dict = total_loss_fn(pred, target)
    
    print(f"Total loss: {total_loss.item():.4f}")
    for name, loss in loss_dict.items():
        if isinstance(loss, torch.Tensor):
            print(f"{name}: {loss.item():.4f}")