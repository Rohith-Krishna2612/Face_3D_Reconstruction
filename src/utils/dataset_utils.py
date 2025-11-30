"""
Dataset handling utilities for FFHQ and other face datasets.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
from typing import List, Tuple, Optional, Dict
import pandas as pd
from pathlib import Path
import kaggle
import zipfile
import shutil


class FFHQDataset(Dataset):
    """FFHQ Dataset for face restoration training."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 image_size: int = 512,
                 degradation_pipeline=None,
                 transform=None,
                 max_samples: Optional[int] = None):
        """
        Args:
            data_dir: Root directory containing FFHQ images
            split: 'train' or 'val'
            image_size: Target image size
            degradation_pipeline: Pipeline to apply degradations
            transform: Additional transforms
            max_samples: Maximum number of samples to use (for quick training)
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.degradation_pipeline = degradation_pipeline
        self.transform = transform
        self.max_samples = max_samples
        
        # Get all image paths
        self.image_paths = self._get_image_paths()
        
        # Split dataset
        self.image_paths = self._split_dataset()
        
        # Limit samples if specified
        if self.max_samples and len(self.image_paths) > self.max_samples:
            self.image_paths = self.image_paths[:self.max_samples]
            print(f"Limited to {self.max_samples} samples for {split} split")
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def _get_image_paths(self) -> List[str]:
        """Get all image paths from the dataset directory."""
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if Path(file).suffix.lower() in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)
    
    def _split_dataset(self) -> List[str]:
        """Split dataset into train/val."""
        total_images = len(self.image_paths)
        
        # Use 90-10 split
        train_split = int(0.9 * total_images)
        
        if self.split == 'train':
            return self.image_paths[:train_split]
        else:  # val
            return self.image_paths[train_split:]
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        image_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.image_size, self.image_size), 
                          interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor format [0, 1]
        gt_image = image.astype(np.float32) / 255.0
        
        # Apply degradation if pipeline is provided
        if self.degradation_pipeline:
            degraded_np, _ = self.degradation_pipeline.apply_random_degradation(image)
            degraded_image = degraded_np.astype(np.float32) / 255.0
        else:
            degraded_image = gt_image.copy()
        
        # Convert to torch tensors (HWC -> CHW)
        gt_tensor = torch.from_numpy(gt_image).permute(2, 0, 1)
        degraded_tensor = torch.from_numpy(degraded_image).permute(2, 0, 1)
        
        # Normalize to [-1, 1] for training
        gt_tensor = gt_tensor * 2.0 - 1.0
        degraded_tensor = degraded_tensor * 2.0 - 1.0
        
        # Apply additional transforms
        if self.transform:
            gt_tensor = self.transform(gt_tensor)
            degraded_tensor = self.transform(degraded_tensor)
        
        return {
            'gt': gt_tensor,
            'lq': degraded_tensor,  # low quality
            'path': image_path
        }


class FaceDatasetDownloader:
    """Utility class to download face datasets."""
    
    @staticmethod
    def download_ffhq_kaggle(save_dir: str = "./data", 
                            dataset_name: str = "rahul18997/ffhq") -> str:
        """Download FFHQ dataset from Kaggle."""
        print("Downloading FFHQ dataset from Kaggle...")
        
        # Ensure kaggle API credentials are set up
        try:
            kaggle.api.authenticate()
        except Exception as e:
            raise RuntimeError(
                "Kaggle API not configured. Please set up your kaggle.json file. "
                f"Error: {e}"
            )
        
        # Create download directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Download dataset
        download_path = os.path.join(save_dir, "ffhq_kaggle")
        os.makedirs(download_path, exist_ok=True)
        
        try:
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=download_path, 
                unzip=True
            )
            
            print(f"FFHQ dataset downloaded to: {download_path}")
            return download_path
            
        except Exception as e:
            raise RuntimeError(f"Failed to download FFHQ dataset: {e}")
    
    @staticmethod
    def setup_ffhq_structure(download_dir: str, target_dir: str) -> str:
        """Reorganize downloaded FFHQ into proper structure."""
        print("Setting up FFHQ directory structure...")
        
        target_path = os.path.join(target_dir, "ffhq")
        os.makedirs(target_path, exist_ok=True)
        
        # Find all images in download directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for root, dirs, files in os.walk(download_dir):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    src_path = os.path.join(root, file)
                    dst_path = os.path.join(target_path, file)
                    
                    # Copy file
                    shutil.copy2(src_path, dst_path)
        
        print(f"FFHQ dataset organized at: {target_path}")
        return target_path


def create_data_loaders(dataset_config: dict, 
                       degradation_pipeline=None) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    data_dir = dataset_config['path']
    batch_size = dataset_config.get('batch_size', 8)
    num_workers = dataset_config.get('num_workers', 4)
    image_size = dataset_config.get('resolution', 512)
    max_train_samples = dataset_config.get('max_train_samples', None)
    max_val_samples = dataset_config.get('max_val_samples', None)
    
    # Create datasets
    train_dataset = FFHQDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        degradation_pipeline=degradation_pipeline,
        max_samples=max_train_samples
    )
    
    val_dataset = FFHQDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        degradation_pipeline=degradation_pipeline,
        max_samples=max_val_samples
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def analyze_dataset(dataset_dir: str) -> Dict:
    """Analyze dataset statistics."""
    print(f"Analyzing dataset: {dataset_dir}")
    
    image_paths = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Get all image paths
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if Path(file).suffix.lower() in valid_extensions:
                image_paths.append(os.path.join(root, file))
    
    total_images = len(image_paths)
    
    if total_images == 0:
        return {'total_images': 0, 'error': 'No images found'}
    
    # Sample a few images to get statistics
    sample_size = min(100, total_images)
    sample_paths = np.random.choice(image_paths, sample_size, replace=False)
    
    sizes = []
    file_sizes = []
    
    for path in sample_paths:
        try:
            # Get image dimensions
            img = cv2.imread(path)
            if img is not None:
                h, w, c = img.shape
                sizes.append((w, h))
            
            # Get file size
            file_size = os.path.getsize(path) / (1024 * 1024)  # MB
            file_sizes.append(file_size)
            
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    # Calculate statistics
    if sizes:
        widths, heights = zip(*sizes)
        stats = {
            'total_images': total_images,
            'sample_analyzed': len(sizes),
            'resolution_stats': {
                'width_range': (min(widths), max(widths)),
                'height_range': (min(heights), max(heights)),
                'width_mean': np.mean(widths),
                'height_mean': np.mean(heights)
            },
            'file_size_stats': {
                'min_mb': min(file_sizes),
                'max_mb': max(file_sizes),
                'mean_mb': np.mean(file_sizes),
                'total_gb': sum(file_sizes) * (total_images / sample_size) / 1024
            }
        }
    else:
        stats = {
            'total_images': total_images,
            'error': 'Could not analyze any images'
        }
    
    return stats


def prepare_dataset_from_kaggle(kaggle_dataset: str, 
                              target_dir: str = "./data") -> str:
    """Complete pipeline to download and prepare dataset from Kaggle."""
    print("Starting dataset preparation pipeline...")
    
    # Download from Kaggle
    downloader = FaceDatasetDownloader()
    download_dir = downloader.download_ffhq_kaggle(target_dir, kaggle_dataset)
    
    # Organize structure
    final_dir = downloader.setup_ffhq_structure(download_dir, target_dir)
    
    # Analyze dataset
    stats = analyze_dataset(final_dir)
    print(f"Dataset analysis: {stats}")
    
    # Cleanup download directory
    if os.path.exists(download_dir) and download_dir != final_dir:
        shutil.rmtree(download_dir)
        print(f"Cleaned up temporary download directory: {download_dir}")
    
    return final_dir


if __name__ == "__main__":
    # Test dataset utilities
    print("Testing dataset utilities...")
    
    # Test dataset analysis on dummy directory
    dummy_dir = "./test_data"
    os.makedirs(dummy_dir, exist_ok=True)
    
    # Create dummy images for testing
    for i in range(5):
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(dummy_dir, f"test_{i}.jpg"), dummy_img)
    
    # Analyze dataset
    stats = analyze_dataset(dummy_dir)
    print(f"Dataset stats: {stats}")
    
    # Clean up
    shutil.rmtree(dummy_dir)
    
    print("Dataset utilities test completed!")