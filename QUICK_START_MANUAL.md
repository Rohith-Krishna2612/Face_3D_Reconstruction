# 🚀 Quick Start Guide - Face Restoration with CodeFormer

## Step-by-Step Setup & Training

### Prerequisites
- Windows with RTX 3050 GPU (4GB VRAM)
- Python 3.8+ installed
- Kaggle account (for dataset download)

---

## 📥 Step 1: Initial Setup (5 minutes)

### 1.1 Create Virtual Environment
```bash
# Navigate to project directory
cd "c:\Users\DELL\Documents\Rohith\Studies\Projects\Face_3D_Construction_Restoration\Face_3D_Reconstruction"

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

### 1.2 Install Dependencies
```bash
# Install all required packages (will auto-detect CUDA)
python setup.py
```

**Expected Output:**
```
Checking for CUDA...
✅ CUDA detected! Installing PyTorch with CUDA 11.8 support
Installing PyTorch with CUDA 11.8...
...
✅ All dependencies installed successfully!
```

### 1.3 Verify GPU Detection
```bash
# Check if PyTorch can see your GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output:**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050
```

---

## 📦 Step 2: Download Dataset (30-60 minutes)

### Download from Kaggle Website
1. Go to: https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
2. Click **"Download"** button (you'll need to sign in)
3. Save the ZIP file (~13 GB)
4. This may take 30-60 minutes depending on your connection

### Extract Dataset
```bash
# Create data directory
mkdir data\ffhq

# Extract the downloaded ZIP file to data\ffhq\
# You can use Windows Explorer (right-click → Extract All)
# Or use PowerShell:
Expand-Archive -Path "Downloads\archive.zip" -DestinationPath "data\ffhq"
```

### Organize Files (if needed)
If images are in subdirectories, move them to `data\ffhq\`:
```bash
# Move all PNG files to data\ffhq\
Get-ChildItem -Path "data\ffhq" -Recurse -Filter "*.png" | Move-Item -Destination "data\ffhq"
```

### Expected Structure:
```
data/
└── ffhq/
    ├── 00000.png
    ├── 00001.png
    ├── 00002.png
    ├── ...
    └── 69999.png
```

### Verify Dataset
```bash
# Count images
(Get-ChildItem -Path "data\ffhq" -Filter "*.png").Count
# Should show around 70,000 images
```

---

## 🧪 Step 3: Quick Test (15 minutes)

### 3.1 Verify Everything Works
```bash
# Run a quick 5-epoch test with 1000 samples
python quick_train.py --epochs 5 --max-samples 1000 --estimate-only
```

**Expected Output:**
```
🚀 Starting CodeFormer Training (RTX 3050 Optimized)
================================================================================
Device: cuda
Mixed Precision: True
Gradient Accumulation Steps: 4
Effective Batch Size: 8
Training for 5 epochs
================================================================================
✅ Mixed precision (FP16) training enabled - saves ~40% VRAM!
✅ Gradient accumulation enabled: 4 steps (simulates batch size of 8)
Loading samples: 1000 images for train split
Loaded 900 images for train split
Loaded 100 images for val split

Epoch 1/5: 100%|████████| 450/450 [06:32<00:00, G_loss: 0.3421, D_loss: 0.1234, VRAM: 3.2GB]
```

---

## 🎯 Step 4: Start Real Training (7 hours)

### 4.1 Train with Good Settings
```bash
# Train with 5000 samples for 20 epochs
python quick_train.py --epochs 20 --max-samples 5000
```

**What Happens:**
- Training: ~6.5 hours
- Validation: ~30 minutes
- Total: ~7 hours
- Checkpoints saved every 5 epochs
- Best model auto-saved

**Monitor Progress:**
```bash
# Open another terminal and run:
tensorboard --logdir logs/codeformer

# Then open browser: http://localhost:6006
```

**What to Watch:**
- `train/g_total` - Should decrease over time
- `val/psnr` - Should increase (higher is better)
- `val/ssim` - Should increase (closer to 1.0 is better)
- `system/gpu_memory_gb` - Should stay under 3.5 GB

---

## 🖥️ Step 5: Start Web Interface (5 minutes)

### 5.1 Start Backend Server
```bash
# Open a new terminal
cd backend
python main.py
```

**Expected Output:**
```
Loading CodeFormer model...
✅ Model loaded from epoch 20
✅ Model loaded successfully!
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 5.2 Start Frontend
```bash
# Open another terminal
cd frontend
npm install
npm start
```

**Expected Output:**
```
Compiled successfully!
You can now view face-restoration-frontend in the browser.
  Local:            http://localhost:3000
```

### 5.3 Open Website
Open browser to: **http://localhost:3000**

---

## 🎨 How the Website Works

### What You'll See:
```
┌─────────────────────────────────────────────────────────────┐
│                  Face Restoration Demo                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Drop image here or click to upload]                       │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│  Results:                                                     │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  ORIGINAL  │  │  BLURRED   │  │  RESTORED  │            │
│  │  RESTORED  │  │  DEGRADED  │  │  FROM BLUR │            │
│  │   IMAGE    │  │   IMAGE    │  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  ORIGINAL  │  │   NOISY    │  │  RESTORED  │            │
│  │  RESTORED  │  │  DEGRADED  │  │  FROM NOISE│            │
│  │   IMAGE    │  │   IMAGE    │  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  ORIGINAL  │  │ COMPRESSED │  │  RESTORED  │            │
│  │  RESTORED  │  │  DEGRADED  │  │FROM COMPRESS│            │
│  │   IMAGE    │  │   IMAGE    │  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│                                                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  ORIGINAL  │  │DOWNSAMPLED │  │  RESTORED  │            │
│  │  RESTORED  │  │  DEGRADED  │  │FROM DOWNSAMP│            │
│  │   IMAGE    │  │   IMAGE    │  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Comparison Layout (What User Requested):
**For each degradation type, you see 3 images side by side:**
1. **Left**: Original restored image (from clean input)
2. **Middle**: Degraded version of the image
3. **Right**: Restored image from the degraded version

**This layout repeats 4 times (one row per degradation type):**
- Row 1: Blur comparison
- Row 2: Noise comparison
- Row 3: JPEG compression comparison
- Row 4: Downsampling comparison

**Purpose**: Shows how well the model handles different types of degradation compared to the clean restoration.

---

## ⏱️ Complete Timeline

### Day 1 (Setup & Test)
```
09:00 - Setup environment (5 min)
09:05 - Install dependencies (10 min)
09:15 - Download dataset (15 min)
09:30 - Quick test training (15 min)
09:45 - Start real training (leave running)
```

### Day 1-2 (Training)
```
09:45 Day 1 - Training starts
16:45 Day 1 - Training ends (~7 hours)
```

### Day 2 (Deploy & Use)
```
16:45 - Check training results
17:00 - Start backend server
17:05 - Start frontend
17:10 - Test with your images!
```

---

## 📊 Training Options Explained

### Option 1: Ultra-Quick Test (15 min)
```bash
python quick_train.py --epochs 5 --max-samples 1000
```
- **Purpose**: Verify setup works
- **Quality**: Basic (just for testing)
- **Use**: First-time setup verification

### Option 2: Good Results (7 hours) ⭐ RECOMMENDED
```bash
python quick_train.py --epochs 20 --max-samples 5000
```
- **Purpose**: Good restoration quality
- **Quality**: Suitable for demos and real use
- **Use**: Your main training run

### Option 3: Best Results (3 days)
```bash
python train.py
```
- **Purpose**: Best possible quality
- **Quality**: Publication-grade restoration
- **Use**: If you have time and need best quality

---

## 🎯 Using Your Trained Model

### Test with Command Line
```bash
# Restore a single image
python inference.py --input test_image.jpg --output restored.jpg --weight 0.5
```

### Test with Web Interface
1. Open: http://localhost:3000
2. Upload your face image
3. See comparisons:
   - Original restoration vs Blur degradation & restoration
   - Original restoration vs Noise degradation & restoration
   - Original restoration vs JPEG degradation & restoration
   - Original restoration vs Downsample degradation & restoration

---

## 🔍 What Each File Does

### Core Files:
- **`setup.py`**: Installs all dependencies with CUDA
- **`config.yaml`**: All training settings (batch size, epochs, etc.)
- **`train.py`**: Main training script
- **`quick_train.py`**: Easy training with presets

### Training:
- **`src/training/train_codeformer.py`**: Training loop with mixed precision
- **`src/models/codeformer.py`**: Model architecture
- **`src/models/losses.py`**: Loss functions
- **`src/degradations/`**: Degradation functions

### Web App:
- **`backend/main.py`**: FastAPI server
- **`frontend/src/`**: React website

---

## 🛠️ Troubleshooting

### Issue 1: "CUDA not available"
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: "Out of memory"
Edit `config.yaml`:
```yaml
training:
  batch_size: 1  # Reduce from 2 to 1
  gradient_accumulation_steps: 8  # Increase from 4 to 8
```

### Issue 3: "Dataset not found"
```bash
# Check if dataset exists
ls data/ffhq/

# If empty, download from Kaggle website
# https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
```

### Issue 4: "Training too slow"
```bash
# Use fewer samples
python quick_train.py --epochs 10 --max-samples 2000
```

### Issue 5: "Port 8000 already in use"
```bash
# Kill existing process
Get-Process -Name "python" | Stop-Process -Force

# Or use different port in backend/main.py
```

---

## 📈 Expected Results

### After 5 epochs (15 min test):
- PSNR: ~20 dB
- SSIM: ~0.65
- Quality: Can barely restore faces
- Status: Just warming up

### After 20 epochs (7 hours):
- PSNR: ~25-28 dB
- SSIM: ~0.80-0.85
- Quality: Good restoration, decent details
- Status: **Ready for use!**

### After 50 epochs (3 days):
- PSNR: ~30+ dB
- SSIM: ~0.90+
- Quality: Excellent restoration, sharp details
- Status: Publication quality

---

## 💡 Tips & Best Practices

### 1. Save Time
- Start with 5k samples, not full 70k
- Use `--estimate-only` first to check timing
- Monitor first few epochs, then let it run

### 2. Save VRAM
- Keep batch_size at 2
- Don't increase resolution above 256
- Close other GPU applications

### 3. Better Results
- Train longer (more epochs)
- Use more data (remove max_samples limit)
- Adjust restoration weight `w` (0.0-1.0)

### 4. Monitor Training
```bash
# Watch TensorBoard
tensorboard --logdir logs/codeformer

# Check sample images
explorer output\samples

# Check GPU usage
nvidia-smi -l 1
```

---

## 🎓 Understanding the Output

### What the Model Does:

1. **Takes your input image**
2. **Applies restoration** (removes blur, noise, etc.)
3. **For demo: Creates 4 degraded versions**
   - Blur version
   - Noisy version
   - Compressed version
   - Downsampled version
4. **Restores each degraded version**
5. **Shows you comparisons**:
   - Original restoration (baseline)
   - Each degradation
   - Restoration from each degradation

### Why This Comparison?
**Shows the model's robustness!**
- Can it restore blurred faces?
- Can it remove noise?
- Can it fix compression artifacts?
- Can it upscale low-resolution faces?

---

## 📞 Need Help?

### Check These First:
1. **GPU Status**: `nvidia-smi`
2. **CUDA in PyTorch**: `python -c "import torch; print(torch.cuda.is_available())"`
3. **Training Logs**: `logs/codeformer/`
4. **Sample Outputs**: `output/samples/`

### Common Questions:

**Q: How long should I train?**
A: 20 epochs with 5k samples = 7 hours. Good starting point!

**Q: Can I stop and resume training?**
A: Yes! Use: `python train.py --resume checkpoints/codeformer/latest_checkpoint.pth`

**Q: How do I know if it's working?**
A: Check `output/samples/` - you should see improving quality over time.

**Q: What's a good PSNR/SSIM?**
A: PSNR > 25 dB and SSIM > 0.80 means it's working well!

---

## ✅ Checklist

### Before Training:
- [ ] Virtual environment activated
- [ ] Dependencies installed (ran `setup.py`)
- [ ] CUDA detected (`torch.cuda.is_available()` returns True)
- [ ] Dataset downloaded (13 GB in `data/ffhq/`)
- [ ] Config file reviewed (`config.yaml`)

### During Training:
- [ ] GPU memory stays under 3.5 GB
- [ ] Loss is decreasing
- [ ] Sample images improving
- [ ] No error messages
- [ ] TensorBoard monitoring active

### After Training:
- [ ] Best checkpoint saved
- [ ] PSNR > 25 dB achieved
- [ ] SSIM > 0.80 achieved
- [ ] Sample quality looks good
- [ ] Ready to test web interface

---

## 🚀 Quick Command Reference

```bash
# Setup
python -m venv venv
.\venv\Scripts\Activate.ps1
python setup.py

# Download data manually from Kaggle
# https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
# Extract to data\ffhq\

# Train
python quick_train.py --epochs 20 --max-samples 5000

# Monitor
tensorboard --logdir logs/codeformer

# Run web app
cd backend && python main.py
cd frontend && npm start

# Test inference
python inference.py --input test.jpg --output restored.jpg
```

---

## 🎉 You're Ready!

**Start with these commands:**
```bash
# 1. Setup (5 min)
python setup.py

# 2. Download dataset manually from Kaggle (30-60 min)
# https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq
# Extract to data\ffhq\

# 3. Quick test (15 min)
python quick_train.py --epochs 5 --max-samples 1000

# 4. Real training (7 hours)
python quick_train.py --epochs 20 --max-samples 5000
```

**After training, launch the web app and start restoring faces! 🎨✨**
