"""
Utility functions for face restoration project.
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(image_path: str, target_size: Tuple[int, int] = None) -> np.ndarray:
    """Load and preprocess image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize if target size specified
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    return image


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save image to disk."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Save image
    success = cv2.imwrite(save_path, image_bgr)
    if not success:
        raise ValueError(f"Failed to save image: {save_path}")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert PyTorch tensor to numpy array."""
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    # Move to CPU
    tensor = tensor.cpu()
    
    # Convert to numpy
    if len(tensor.shape) == 4:  # Batch dimension
        tensor = tensor.squeeze(0)
    
    # CHW to HWC
    if len(tensor.shape) == 3:
        tensor = tensor.permute(1, 2, 0)
    
    # Convert to numpy and scale to [0, 255]
    numpy_array = tensor.numpy()
    
    # Assuming tensor is in [-1, 1] or [0, 1]
    if numpy_array.min() < 0:
        numpy_array = (numpy_array + 1) / 2  # [-1, 1] -> [0, 1]
    
    numpy_array = np.clip(numpy_array * 255, 0, 255).astype(np.uint8)
    
    return numpy_array


def numpy_to_tensor(array: np.ndarray, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Convert numpy array to PyTorch tensor."""
    # Normalize to [0, 1]
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    
    # HWC to CHW
    if len(array.shape) == 3:
        array = np.transpose(array, (2, 0, 1))
    
    # Convert to tensor
    tensor = torch.from_numpy(array).to(device)
    
    # Add batch dimension if needed
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    
    return tensor


def create_comparison_grid(images: List[np.ndarray], 
                         titles: List[str] = None,
                         save_path: str = None) -> np.ndarray:
    """Create a comparison grid of images."""
    if titles is None:
        titles = [f"Image {i+1}" for i in range(len(images))]
    
    n_images = len(images)
    n_cols = min(n_images, 4)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot images
    for i, (img, title) in enumerate(zip(images, titles)):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].set_title(title)
            axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    
    return buf


def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Calculate image quality metrics."""
    
    def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images."""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images."""
        try:
            from skimage.metrics import structural_similarity
            if len(img1.shape) == 3:
                return structural_similarity(img1, img2, multichannel=True, channel_axis=2, data_range=1.0)
            else:
                return structural_similarity(img1, img2, data_range=1.0)
        except ImportError:
            # Fallback if scikit-image not available
            return 0.0
    
    # Convert to same data type
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    
    metrics = {
        'psnr': psnr(pred, target),
        'ssim': ssim(pred, target),
        'mae': np.mean(np.abs(pred - target)),
        'mse': np.mean((pred - target) ** 2)
    }
    
    return metrics


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int], 
                    interpolation: str = 'linear') -> np.ndarray:
        """Resize image with specified interpolation."""
        interp_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        interp = interp_map.get(interpolation, cv2.INTER_LINEAR)
        return cv2.resize(image, size, interpolation=interp)
    
    @staticmethod
    def crop_center(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Crop center region of image."""
        h, w = image.shape[:2]
        crop_h, crop_w = size
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    @staticmethod
    def normalize_image(image: np.ndarray, 
                       mean: List[float] = [0.5, 0.5, 0.5],
                       std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
        """Normalize image with mean and std."""
        image = image.astype(np.float32) / 255.0
        
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray,
                         mean: List[float] = [0.5, 0.5, 0.5],
                         std: List[float] = [0.5, 0.5, 0.5]) -> np.ndarray:
        """Denormalize image."""
        if len(image.shape) == 3:
            for i in range(3):
                image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image


def setup_directories(base_path: str) -> Dict[str, str]:
    """Setup project directories."""
    directories = {
        'data': os.path.join(base_path, 'data'),
        'models': os.path.join(base_path, 'models'),
        'checkpoints': os.path.join(base_path, 'checkpoints'),
        'logs': os.path.join(base_path, 'logs'),
        'output': os.path.join(base_path, 'output'),
        'input': os.path.join(base_path, 'data', 'input'),
        'enhanced': os.path.join(base_path, 'data', 'enhanced'),
        'output_3d': os.path.join(base_path, 'data', 'output_3d')
    }
    
    # Create directories
    for name, path in directories.items():
        os.makedirs(path, exist_ok=True)
        print(f"Created directory: {path}")
    
    return directories


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   save_path: str,
                   **kwargs) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved: {save_path}")


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   checkpoint_path: str) -> Dict:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    
    return checkpoint


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Test tensor conversion
    tensor = numpy_to_tensor(dummy_image)
    converted_back = tensor_to_numpy(tensor)
    
    print(f"Original shape: {dummy_image.shape}")
    print(f"Tensor shape: {tensor.shape}")
    print(f"Converted back shape: {converted_back.shape}")
    
    # Test metrics
    metrics = calculate_metrics(dummy_image, converted_back)
    print(f"Metrics: {metrics}")
    
    print("Utility functions test completed!")