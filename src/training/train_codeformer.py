"""
Training script for CodeFormer face restoration model.
Optimized for RTX 3050 with 4GB VRAM.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import argparse
from typing import Dict, Any
import time
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.models.codeformer import create_codeformer_model
from src.models.discriminator import create_discriminator
from src.models.losses import create_loss_functions, AdversarialLoss
from src.degradations import create_degradation_pipeline
from src.utils import (
    create_data_loaders, setup_directories, get_device,
    save_checkpoint, load_checkpoint, tensor_to_numpy, 
    save_image, calculate_metrics
)


class CodeFormerTrainer:
    """Trainer class for CodeFormer model optimized for RTX 3050."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        self.config = self.load_config(config_path)
        self.device = get_device()
        self.setup_directories()
        self.setup_logging()
        self.setup_models()
        self.setup_optimizers()
        self.setup_data()
        self.setup_losses()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.step = 0
        
        # RTX 3050 optimizations
        self.mixed_precision = self.config['training'].get('mixed_precision', False)
        self.gradient_accumulation_steps = self.config['training'].get('gradient_accumulation_steps', 1)
        
        # Setup mixed precision training
        if self.mixed_precision:
            self.scaler_g = torch.cuda.amp.GradScaler()
            self.scaler_d = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision (FP16) training enabled - saves ~40% VRAM!")
        
        if self.gradient_accumulation_steps > 1:
            print(f"✅ Gradient accumulation enabled: {self.gradient_accumulation_steps} steps (simulates batch size of {self.config['training']['batch_size'] * self.gradient_accumulation_steps})")
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def setup_directories(self):
        """Setup training directories."""
        # Use paths from config if specified (for Colab/Drive), otherwise use project root
        base_paths = self.config.get('paths', {})
        
        if 'checkpoints' in base_paths and base_paths['checkpoints'].startswith('/content/drive'):
            # Using Google Drive paths - don't call setup_directories, use config directly
            self.checkpoint_dir = os.path.join(base_paths['checkpoints'], 'codeformer')
            self.log_dir = os.path.join(base_paths['logs'], 'codeformer')
            self.sample_dir = os.path.join(base_paths['output'], 'samples')
            
            print(f"✅ Using Google Drive storage:")
            print(f"   Checkpoints: {self.checkpoint_dir}")
            print(f"   Logs: {self.log_dir}")
            print(f"   Samples: {self.sample_dir}")
        else:
            # Local training - use project root
            self.dirs = setup_directories(os.getcwd())
            self.checkpoint_dir = os.path.join(self.dirs['checkpoints'], 'codeformer')
            self.log_dir = os.path.join(self.dirs['logs'], 'codeformer')
            self.sample_dir = os.path.join(self.dirs['output'], 'samples')
        
        # Create the final directories
        for directory in [self.checkpoint_dir, self.log_dir, self.sample_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging and monitoring."""
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # Weights & Biases (optional)
        if self.config.get('use_wandb', False):
            wandb.init(
                project="codeformer-face-restoration",
                config=self.config,
                name=f"codeformer_rtx3050_{int(time.time())}"
            )
    
    def setup_models(self):
        """Setup models."""
        # Generator (CodeFormer)
        self.generator = create_codeformer_model(self.config).to(self.device)
        
        # Discriminator
        self.discriminator = create_discriminator(self.config).to(self.device)
        
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def setup_optimizers(self):
        """Setup optimizers."""
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        self.optimizer_g = optim.AdamW(
            self.generator.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.optimizer_d = optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, 
            T_max=self.config['training']['num_epochs']
        )
        
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d,
            T_max=self.config['training']['num_epochs']
        )
    
    def setup_data(self):
        """Setup data loaders."""
        # Create degradation pipeline
        self.degradation_pipeline = create_degradation_pipeline(self.config)
        
        # Create data loaders
        dataset_config = {
            'path': self.config['dataset']['path'],
            'batch_size': self.config['training']['batch_size'],
            'num_workers': self.config['training'].get('num_workers', 2),
            'resolution': self.config['dataset']['resolution']
        }
        
        # Add max samples for limited training
        if self.config['dataset'].get('max_train_samples'):
            dataset_config['max_train_samples'] = self.config['dataset']['max_train_samples']
        if self.config['dataset'].get('max_val_samples'):
            dataset_config['max_val_samples'] = self.config['dataset']['max_val_samples']
        
        self.train_loader, self.val_loader = create_data_loaders(
            dataset_config, 
            self.degradation_pipeline
        )
        
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
    
    def setup_losses(self):
        """Setup loss functions."""
        self.criterion_g = create_loss_functions(self.config)
        self.criterion_d = AdversarialLoss('lsgan')
        
        self.criterion_g.to(self.device)
        self.criterion_d.to(self.device)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with RTX 3050 optimizations."""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {'g_total': 0, 'd_real': 0, 'd_fake': 0}
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            lq_images = batch['lq'].to(self.device)  # low quality
            gt_images = batch['gt'].to(self.device)  # ground truth
            
            batch_size = lq_images.size(0)
            
            # ===================
            # Train Discriminator
            # ===================
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    # Real images
                    pred_real = self.discriminator(gt_images)
                    loss_d_real = self.criterion_d(pred_real, True)
                    
                    # Fake images
                    with torch.no_grad():
                        fake_images, _ = self.generator(lq_images, w=0.5)
                    pred_fake = self.discriminator(fake_images.detach())
                    loss_d_fake = self.criterion_d(pred_fake, False)
                    
                    # Total discriminator loss
                    loss_d = (loss_d_real + loss_d_fake) * 0.5
                
                self.optimizer_d.zero_grad()
                self.scaler_d.scale(loss_d).backward()
                self.scaler_d.step(self.optimizer_d)
                self.scaler_d.update()
            else:
                self.optimizer_d.zero_grad()
                
                # Real images
                pred_real = self.discriminator(gt_images)
                loss_d_real = self.criterion_d(pred_real, True)
                
                # Fake images
                with torch.no_grad():
                    fake_images, _ = self.generator(lq_images, w=0.5)
                pred_fake = self.discriminator(fake_images.detach())
                loss_d_fake = self.criterion_d(pred_fake, False)
                
                # Total discriminator loss
                loss_d = (loss_d_real + loss_d_fake) * 0.5
                loss_d.backward()
                self.optimizer_d.step()
            
            # ===================
            # Train Generator
            # ===================
            if self.mixed_precision:
                with torch.amp.autocast('cuda'):
                    # Generate restored images
                    restored_images, info = self.generator(lq_images, w=0.5)
                    
                    # Adversarial loss
                    pred_fake = self.discriminator(restored_images)
                    
                    # Combined loss
                    loss_g, loss_dict = self.criterion_g(
                        restored_images, 
                        gt_images,
                        pred_fake,
                        info.get('vq_loss', None)
                    )
                    
                    # Scale loss for gradient accumulation
                    loss_g = loss_g / self.gradient_accumulation_steps
                
                self.scaler_g.scale(loss_g).backward()
                
                # Only step optimizer after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler_g.step(self.optimizer_g)
                    self.scaler_g.update()
                    self.optimizer_g.zero_grad()
            else:
                # Generate restored images
                restored_images, info = self.generator(lq_images, w=0.5)
                
                # Adversarial loss
                pred_fake = self.discriminator(restored_images)
                
                # Combined loss
                loss_g, loss_dict = self.criterion_g(
                    restored_images, 
                    gt_images,
                    pred_fake,
                    info.get('vq_loss', None)
                )
                
                # Scale loss for gradient accumulation
                loss_g = loss_g / self.gradient_accumulation_steps
                loss_g.backward()
                
                # Only step optimizer after accumulation steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer_g.step()
                    self.optimizer_g.zero_grad()
            
            # Update metrics (multiply back by accumulation steps for logging)
            epoch_losses['g_total'] += loss_g.item() * self.gradient_accumulation_steps
            epoch_losses['d_real'] += loss_d_real.item()
            epoch_losses['d_fake'] += loss_d_fake.item()
            
            # Update progress bar with VRAM monitoring
            pbar.set_postfix({
                'G_loss': f"{loss_g.item() * self.gradient_accumulation_steps:.4f}",
                'D_loss': f"{loss_d.item():.4f}",
                'VRAM': f"{torch.cuda.memory_allocated()/1024**3:.1f}GB"
            })
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.log_training_step(loss_dict, loss_d_real, loss_d_fake)
            
            # Save samples
            if batch_idx % 500 == 0:
                self.save_training_samples(lq_images, restored_images, gt_images, epoch, batch_idx)
            
            self.step += 1
        
        # Make sure to step optimizer if there are remaining accumulated gradients
        if self.mixed_precision and (len(self.train_loader) % self.gradient_accumulation_steps != 0):
            self.scaler_g.step(self.optimizer_g)
            self.scaler_g.update()
            self.optimizer_g.zero_grad()
        elif not self.mixed_precision and (len(self.train_loader) % self.gradient_accumulation_steps != 0):
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.generator.eval()
        
        val_losses = {'total': 0, 'psnr': 0, 'ssim': 0}
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validation")):
                lq_images = batch['lq'].to(self.device)
                gt_images = batch['gt'].to(self.device)
                
                # Use mixed precision for validation too
                if self.mixed_precision:
                    with torch.amp.autocast('cuda'):
                        # Generate restored images
                        restored_images, info = self.generator(lq_images, w=0.5)
                        
                        # Calculate loss
                        loss_g, loss_dict = self.criterion_g(restored_images, gt_images, vq_loss=info.get('vq_loss', None))
                else:
                    # Generate restored images
                    restored_images, info = self.generator(lq_images, w=0.5)
                    
                    # Calculate loss
                    loss_g, loss_dict = self.criterion_g(restored_images, gt_images, vq_loss=info.get('vq_loss', None))
                
                val_losses['total'] += loss_g.item()
                
                # Calculate metrics (convert to numpy for PSNR/SSIM)
                if batch_idx < 10:  # Only calculate for first few batches to save time
                    restored_np = tensor_to_numpy(restored_images[0])
                    gt_np = tensor_to_numpy(gt_images[0])
                    
                    metrics = calculate_metrics(restored_np, gt_np)
                    val_losses['psnr'] += metrics['psnr']
                    val_losses['ssim'] += metrics['ssim']
        
        # Average losses
        for key in val_losses:
            if key in ['psnr', 'ssim']:
                val_losses[key] /= min(10, num_batches)
            else:
                val_losses[key] /= num_batches
        
        return val_losses
    
    def log_training_step(self, loss_dict: Dict, d_real: float, d_fake: float):
        """Log training step metrics."""
        # TensorBoard logging
        for name, loss in loss_dict.items():
            if isinstance(loss, torch.Tensor):
                self.writer.add_scalar(f'train/{name}', loss.item(), self.step)
        
        self.writer.add_scalar('train/d_real', d_real, self.step)
        self.writer.add_scalar('train/d_fake', d_fake, self.step)
        self.writer.add_scalar('system/gpu_memory_gb', torch.cuda.memory_allocated()/1024**3, self.step)
        
        # Weights & Biases logging
        if hasattr(self, 'wandb') and wandb.run is not None:
            log_dict = {f'train/{k}': v.item() if isinstance(v, torch.Tensor) else v 
                       for k, v in loss_dict.items()}
            log_dict.update({
                'train/d_real': d_real,
                'train/d_fake': d_fake,
                'step': self.step,
                'gpu_memory_gb': torch.cuda.memory_allocated()/1024**3
            })
            wandb.log(log_dict)
    
    def save_training_samples(self, lq: torch.Tensor, restored: torch.Tensor, 
                            gt: torch.Tensor, epoch: int, batch_idx: int):
        """Save training samples."""
        # Convert first sample in batch to numpy
        lq_np = tensor_to_numpy(lq[0])
        restored_np = tensor_to_numpy(restored[0])
        gt_np = tensor_to_numpy(gt[0])
        
        # Create comparison
        comparison = np.hstack([lq_np, restored_np, gt_np])
        
        # Save
        save_path = os.path.join(self.sample_dir, f"epoch_{epoch:03d}_batch_{batch_idx:05d}.jpg")
        save_image(comparison, save_path)
    
    def save_model_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Add scaler state for mixed precision
        if self.mixed_precision:
            checkpoint_data['scaler_g_state_dict'] = self.scaler_g.state_dict()
            checkpoint_data['scaler_d_state_dict'] = self.scaler_d.state_dict()
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint_data, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint_data, best_path)
            print(f"🌟 New best model saved with validation loss: {val_loss:.4f}")
        
        # Save epoch checkpoint
        if epoch % self.config['training']['save_freq'] == 0:
            epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save(checkpoint_data, epoch_path)
    
    def load_from_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        self.scheduler_d.load_state_dict(checkpoint['scheduler_d_state_dict'])
        
        # Load scaler state for mixed precision
        if self.mixed_precision and 'scaler_g_state_dict' in checkpoint:
            self.scaler_g.load_state_dict(checkpoint['scaler_g_state_dict'])
            self.scaler_d.load_state_dict(checkpoint['scaler_d_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        print(f"✅ Resumed from epoch {self.current_epoch} with validation loss: {self.best_val_loss:.4f}")
    
    def train(self, resume_from_checkpoint: str = None):
        """Main training loop."""
        if resume_from_checkpoint:
            self.load_from_checkpoint(resume_from_checkpoint)
        
        print("=" * 80)
        print("🚀 Starting CodeFormer Training (RTX 3050 Optimized)")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Gradient Accumulation Steps: {self.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.config['training']['batch_size'] * self.gradient_accumulation_steps}")
        print(f"Training for {self.config['training']['num_epochs']} epochs")
        print("=" * 80)
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            # Training
            train_losses = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config['training']['val_freq'] == 0:
                val_losses = self.validate_epoch(epoch)
                
                # Log validation metrics
                print(f"\n{'='*80}")
                print(f"Epoch {epoch+1}/{self.config['training']['num_epochs']}:")
                print(f"  Train Loss: {train_losses['g_total']:.4f}")
                print(f"  Val Loss: {val_losses['total']:.4f}")
                print(f"  Val PSNR: {val_losses['psnr']:.2f} dB")
                print(f"  Val SSIM: {val_losses['ssim']:.4f}")
                print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
                print("="*80)
                
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats()
                
                # TensorBoard logging
                self.writer.add_scalar('val/total_loss', val_losses['total'], epoch)
                self.writer.add_scalar('val/psnr', val_losses['psnr'], epoch)
                self.writer.add_scalar('val/ssim', val_losses['ssim'], epoch)
                
                # Check if best model
                is_best = val_losses['total'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_losses['total']
                
                # Save checkpoint
                self.save_model_checkpoint(epoch, val_losses['total'], is_best)
            
            # Update learning rates
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            self.current_epoch = epoch + 1
        
        print("\n" + "="*80)
        print("🎉 Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*80)
        
        # Close logging
        self.writer.close()
        if hasattr(self, 'wandb') and wandb.run is not None:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train CodeFormer face restoration model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("⚠️  WARNING: No GPU detected! Training will be very slow on CPU.")
        print("   Make sure CUDA is installed correctly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    else:
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    # Create trainer
    trainer = CodeFormerTrainer(args.config)
    
    # Start training
    trainer.train(resume_from_checkpoint=args.resume)


if __name__ == "__main__":
    main()
