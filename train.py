"""
Simple script to start training CodeFormer model.
Run this after setting up your dataset and virtual environment.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.utils.dataset_utils import prepare_dataset_from_kaggle, analyze_dataset
from src.training.train_codeformer import CodeFormerTrainer


def setup_dataset(config_path: str = "config.yaml"):
    """Setup dataset from Kaggle or analyze existing dataset."""
    print("Setting up dataset...")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_path = config['dataset']['path']
    
    # Check if dataset already exists
    if os.path.exists(dataset_path):
        print(f"Dataset directory found: {dataset_path}")
        stats = analyze_dataset(dataset_path)
        print(f"Dataset statistics: {stats}")
        
        if stats.get('total_images', 0) == 0:
            print("Dataset directory is empty!")
            download = input("Would you like to download FFHQ from Kaggle? (y/n): ").lower() == 'y'
            if download:
                kaggle_dataset = input("Enter Kaggle dataset name (default: rahul18997/ffhq): ").strip()
                if not kaggle_dataset:
                    kaggle_dataset = "rahul18997/ffhq"
                
                try:
                    final_dir = prepare_dataset_from_kaggle(kaggle_dataset, "./data")
                    
                    # Update config with correct path
                    config['dataset']['path'] = final_dir
                    with open(config_path, 'w') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    print(f"Dataset downloaded and configured: {final_dir}")
                except Exception as e:
                    print(f"Failed to download dataset: {e}")
                    print("Please manually download and extract FFHQ dataset to the configured path.")
                    return False
        return True
    else:
        print(f"Dataset directory not found: {dataset_path}")
        download = input("Would you like to download FFHQ from Kaggle? (y/n): ").lower() == 'y'
        
        if download:
            kaggle_dataset = input("Enter Kaggle dataset name (default: rahul18997/ffhq): ").strip()
            if not kaggle_dataset:
                kaggle_dataset = "rahul18997/ffhq"
            
            try:
                final_dir = prepare_dataset_from_kaggle(kaggle_dataset, "./data")
                
                # Update config with correct path
                config['dataset']['path'] = final_dir
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                print(f"Dataset downloaded and configured: {final_dir}")
                return True
            except Exception as e:
                print(f"Failed to download dataset: {e}")
                print("Please manually download and extract FFHQ dataset.")
                print("Make sure to update the 'dataset.path' in config.yaml")
                return False
        else:
            print("Please manually download and extract FFHQ dataset.")
            print("Update the 'dataset.path' in config.yaml with the correct path.")
            return False


def check_requirements():
    """Check if required packages are installed."""
    try:
        import torch
        import torchvision
        import cv2
        import numpy
        import yaml
        import tqdm
        print("✓ All required packages are available")
        
        # Check CUDA availability
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("! CUDA not available - training will use CPU (slower)")
        
        return True
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train CodeFormer face restoration model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup dataset, don\'t start training')
    parser.add_argument('--skip-setup', action='store_true',
                       help='Skip dataset setup and go straight to training')
    
    args = parser.parse_args()
    
    print("🎯 CodeFormer Training Setup")
    print("=" * 50)
    
    # Check requirements
    print("\n1. Checking requirements...")
    if not check_requirements():
        return
    
    # Setup dataset
    if not args.skip_setup:
        print("\n2. Setting up dataset...")
        if not setup_dataset(args.config):
            print("Dataset setup failed. Please set up the dataset manually.")
            return
    
    if args.setup_only:
        print("\nDataset setup completed! You can now run training with:")
        print(f"python train.py --config {args.config}")
        return
    
    # Start training
    print("\n3. Starting training...")
    try:
        trainer = CodeFormerTrainer(args.config)
        trainer.train(resume_from_checkpoint=args.resume)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        print("Please check your configuration and dataset setup.")


if __name__ == "__main__":
    main()