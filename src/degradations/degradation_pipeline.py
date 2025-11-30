"""
Degradation functions for face restoration training.
Includes blur, gaussian noise, JPEG compression, and downsampling.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import io
import random
from typing import Union, Tuple, List


class DegradationPipeline:
    """Pipeline for applying various degradations to face images."""
    
    def __init__(self, config: dict):
        self.config = config
        
    def apply_blur(self, image: np.ndarray, kernel_size: int = None, sigma: float = None) -> np.ndarray:
        """Apply Gaussian blur to the image."""
        if kernel_size is None:
            kernel_size = random.choice(self.config['degradations']['blur']['kernel_sizes'])
        if sigma is None:
            sigma = random.uniform(*self.config['degradations']['blur']['sigma_range'])
            
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_gaussian_noise(self, image: np.ndarray, noise_level: float = None) -> np.ndarray:
        """Add Gaussian noise to the image."""
        if noise_level is None:
            noise_level = random.uniform(*self.config['degradations']['gaussian_noise']['noise_range'])
            
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def apply_jpeg_compression(self, image: np.ndarray, quality: int = None) -> np.ndarray:
        """Apply JPEG compression degradation."""
        if quality is None:
            quality = random.randint(*self.config['degradations']['jpeg_compression']['quality_range'])
            
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
            
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_image = Image.open(buffer)
        
        # Convert back to numpy array
        result = np.array(compressed_image)
        if len(image.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
        return result
    
    def apply_downsampling(self, image: np.ndarray, scale_factor: int = None) -> np.ndarray:
        """Apply downsampling degradation."""
        if scale_factor is None:
            scale_factor = random.choice(self.config['degradations']['downsampling']['scale_factors'])
            
        h, w = image.shape[:2]
        
        # Downsample
        small_h, small_w = h // scale_factor, w // scale_factor
        downsampled = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
        
        # Upsample back to original size
        upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return upsampled
    
    def apply_random_degradation(self, image: np.ndarray, degradation_type: str = None) -> Tuple[np.ndarray, str]:
        """Apply a random degradation or a specific one."""
        degradation_types = ['blur', 'gaussian_noise', 'jpeg_compression', 'downsampling']
        
        if degradation_type is None:
            degradation_type = random.choice(degradation_types)
            
        if degradation_type == 'blur':
            degraded = self.apply_blur(image)
        elif degradation_type == 'gaussian_noise':
            degraded = self.apply_gaussian_noise(image)
        elif degradation_type == 'jpeg_compression':
            degraded = self.apply_jpeg_compression(image)
        elif degradation_type == 'downsampling':
            degraded = self.apply_downsampling(image)
        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")
            
        return degraded, degradation_type
    
    def apply_multiple_degradations(self, image: np.ndarray, 
                                  degradation_types: List[str] = None) -> dict:
        """Apply multiple degradations and return all results."""
        if degradation_types is None:
            degradation_types = ['blur', 'gaussian_noise', 'jpeg_compression', 'downsampling']
            
        results = {'original': image.copy()}
        
        for deg_type in degradation_types:
            degraded, _ = self.apply_random_degradation(image, deg_type)
            results[deg_type] = degraded
            
        return results
    
    def create_training_pair(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """Create a training pair (degraded, clean) with degradation type."""
        degraded, deg_type = self.apply_random_degradation(image)
        return degraded, image, deg_type


class TensorDegradation:
    """PyTorch tensor-based degradations for training."""
    
    @staticmethod
    def add_gaussian_noise_tensor(tensor: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise to tensor (values in [0, 1])."""
        noise = torch.randn_like(tensor) * noise_std
        return torch.clamp(tensor + noise, 0, 1)
    
    @staticmethod
    def apply_blur_tensor(tensor: torch.Tensor, kernel_size: int = 15, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur to tensor."""
        # Create Gaussian kernel
        kernel = TensorDegradation._get_gaussian_kernel(kernel_size, sigma)
        kernel = kernel.to(tensor.device).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(tensor.shape[1], 1, 1, 1)
        
        # Apply convolution
        blurred = F.conv2d(tensor, kernel, groups=tensor.shape[1], padding=kernel_size//2)
        return blurred
    
    @staticmethod
    def _get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        """Generate 2D Gaussian kernel."""
        x = torch.arange(kernel_size, dtype=torch.float32)
        x = x - kernel_size // 2
        gaussian_1d = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        gaussian_2d = gaussian_1d.unsqueeze(1) * gaussian_1d.unsqueeze(0)
        return gaussian_2d / gaussian_2d.sum()
    
    @staticmethod
    def downsample_tensor(tensor: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        """Downsample and upsample tensor."""
        # Downsample
        downsampled = F.interpolate(tensor, scale_factor=1/scale_factor, mode='area')
        # Upsample back
        upsampled = F.interpolate(downsampled, size=tensor.shape[2:], mode='bilinear', align_corners=False)
        return upsampled


def create_degradation_pipeline(config: dict) -> DegradationPipeline:
    """Factory function to create degradation pipeline."""
    return DegradationPipeline(config)


if __name__ == "__main__":
    # Test degradation pipeline
    import yaml
    
    # Load config
    with open('../../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create pipeline
    pipeline = create_degradation_pipeline(config)
    
    # Test with a dummy image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    # Test all degradations
    results = pipeline.apply_multiple_degradations(test_image)
    
    print("Degradation pipeline test completed successfully!")
    print(f"Generated {len(results)} variations: {list(results.keys())}")