"""
Quick training script with command-line options for easy experimentation.
"""

import argparse
import yaml
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.training.train_codeformer import CodeFormerTrainer


def create_quick_config(base_config_path, args):
    """Create a modified config based on command-line args."""
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with command-line arguments
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    if args.resolution:
        config['dataset']['resolution'] = args.resolution
        config['model']['input_size'] = [args.resolution, args.resolution]
    
    if args.max_samples:
        config['dataset']['max_train_samples'] = args.max_samples
        config['dataset']['max_val_samples'] = args.max_samples // 10
    
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    return config


def estimate_training_time(config):
    """Estimate training time based on configuration."""
    batch_size = config['training']['batch_size']
    epochs = config['training']['num_epochs']
    resolution = config['dataset']['resolution']
    max_samples = config['dataset'].get('max_train_samples', 52000)
    
    if max_samples is None:
        max_samples = 52000  # Assume full FFHQ (52,001 images)
    
    # Rough estimates (adjust based on your GPU)
    batches_per_epoch = max_samples // batch_size
    
    # Time per batch in seconds (very rough estimates for RTX 3050)
    if resolution == 128:
        seconds_per_batch = 0.5
    elif resolution == 256:
        seconds_per_batch = 2.0
    elif resolution == 512:
        seconds_per_batch = 8.0
    else:
        seconds_per_batch = 2.0
    
    total_seconds = batches_per_epoch * epochs * seconds_per_batch
    
    hours = total_seconds / 3600
    days = hours / 24
    
    return hours, days, batches_per_epoch


def main():
    parser = argparse.ArgumentParser(
        description='Train CodeFormer with easy command-line options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (10 minutes):
    python quick_train.py --epochs 2 --max-samples 100 --resolution 128
  
  Small run (7 hours):
    python quick_train.py --epochs 20 --max-samples 5000 --resolution 256
  
  Medium run (18 hours):
    python quick_train.py --epochs 30 --max-samples 10000 --resolution 256
  
  Full training (3-6 days):
    python quick_train.py --epochs 50 --resolution 256
        """
    )
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Base configuration file')
    parser.add_argument('--epochs', type=int,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (be careful with GPU memory!)')
    parser.add_argument('--resolution', type=int, choices=[128, 256, 512],
                       help='Image resolution (lower = faster)')
    parser.add_argument('--max-samples', type=int,
                       help='Maximum training samples (for quick experiments)')
    parser.add_argument('--lr', type=float,
                       help='Learning rate')
    parser.add_argument('--resume', type=str,
                       help='Resume from checkpoint')
    parser.add_argument('--estimate-only', action='store_true',
                       help='Only estimate training time, don\'t train')
    parser.add_argument('--gpu-check', action='store_true',
                       help='Check GPU availability and memory')
    
    args = parser.parse_args()
    
    # GPU check
    if args.gpu_check:
        import torch
        print("\n" + "="*60)
        print("🖥️  GPU Information")
        print("="*60)
        
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: Yes")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            print(f"❌ CUDA Available: No")
            print(f"   Will use CPU (very slow for training!)")
        print("="*60 + "\n")
        
        if not args.estimate_only:
            return
    
    # Create config
    config = create_quick_config(args.config, args)
    
    # Estimate training time
    hours, days, batches = estimate_training_time(config)
    
    print("\n" + "="*60)
    print("⏱️  Training Time Estimate")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Batch Size: {config['training']['batch_size']}")
    print(f"  - Resolution: {config['dataset']['resolution']}x{config['dataset']['resolution']}")
    print(f"  - Max Samples: {config['dataset'].get('max_train_samples', 'Full dataset (~70k)')}")
    print(f"  - Learning Rate: {config['training']['learning_rate']}")
    print()
    print(f"Estimated Training Time:")
    print(f"  - Batches per epoch: ~{batches:,}")
    print(f"  - Total time: ~{hours:.1f} hours ({days:.1f} days)")
    print()
    
    if hours < 1:
        print(f"  ⚡ Very Fast! ({int(hours*60)} minutes)")
    elif hours < 12:
        print(f"  🚀 Quick run! (Half a day)")
    elif hours < 48:
        print(f"  ⏰ Overnight run (1-2 days)")
    else:
        print(f"  ⏳ Long run (Plan accordingly!)")
    
    print("="*60 + "\n")
    
    if args.estimate_only:
        print("Estimate only mode - not starting training.")
        print("Remove --estimate-only to start actual training.")
        return
    
    # Confirm before long training
    if hours > 24:
        response = input(f"⚠️  This will take ~{days:.1f} days. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return
    
    # Save modified config
    temp_config_path = 'config_temp.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"🚀 Starting training...")
    print(f"📝 Using config: {temp_config_path}")
    print(f"💾 Checkpoints will be saved every {config['training']['save_freq']} epochs")
    print(f"📊 Monitor with: tensorboard --logdir logs/")
    print()
    
    try:
        # Create trainer with modified config
        trainer = CodeFormerTrainer(temp_config_path)
        
        # Start training
        trainer.train(resume_from_checkpoint=args.resume)
        
        print("\n✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏸️  Training interrupted by user.")
        print("You can resume later with: --resume checkpoints/codeformer/latest_checkpoint.pth")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup temp config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    main()