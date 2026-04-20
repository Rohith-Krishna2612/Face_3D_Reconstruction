import torch

ckpt = torch.load('checkpoints/codeformer/latest_checkpoint.pth', map_location='cpu')

print('='*60)
print('CHECKPOINT INFO')
print('='*60)
print(f'Epoch: {ckpt["epoch"]}')
print(f'Validation Loss: {ckpt["val_loss"]:.4f}')
print()
print('Training Config from Colab:')
print(f'  Batch size: {ckpt["config"]["training"]["batch_size"]}')
print(f'  Max train samples: {ckpt["config"]["dataset"].get("max_train_samples", "Full dataset")}')
print(f'  Resolution: {ckpt["config"]["dataset"]["resolution"]}')
print(f'  Num epochs: {ckpt["config"]["training"]["num_epochs"]}')
print(f'  Mixed precision: {ckpt["config"]["training"].get("mixed_precision", False)}')
print(f'  Gradient accumulation: {ckpt["config"]["training"].get("gradient_accumulation_steps", 1)}')
print('='*60)
