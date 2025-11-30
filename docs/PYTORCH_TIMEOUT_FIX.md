# 🔧 PyTorch Installation Timeout - Quick Fix

## Problem
PyTorch download is timing out due to slow/unstable internet connection.

---

## ✅ Solution 1: Use Slow Connection Installer (RECOMMENDED)

I've created a special installer for slow connections:

```bash
python setup_slow.py
```

**What it does:**
- Installs packages ONE BY ONE (more reliable)
- Automatic retry with delays
- Better timeout handling
- Real-time progress display
- Fallback options if CUDA fails

**Time:** 15-30 minutes (depends on your connection)

---

## ✅ Solution 2: Manual Installation (If Solution 1 Fails)

### Step 1: Install torch only (largest package)
```bash
pip install --default-timeout=300 torch --index-url https://download.pytorch.org/whl/cu118
```

Wait for it to complete (may take 10-15 minutes).

### Step 2: Install torchvision
```bash
pip install --default-timeout=300 torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Install torchaudio
```bash
pip install --default-timeout=300 torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 4: Install other requirements
```bash
pip install -r requirements.txt
```

---

## ✅ Solution 3: Download Pre-built Wheels (Most Reliable)

If your connection keeps timing out, download the wheel files manually:

### Step 1: Download Wheel Files

Visit: https://download.pytorch.org/whl/torch_stable.html

Find and download these files for **Python 3.11** and **CUDA 11.8**:

1. **torch**: `torch-2.1.0+cu118-cp311-cp311-win_amd64.whl` (~2.4 GB)
2. **torchvision**: `torchvision-0.16.0+cu118-cp311-cp311-win_amd64.whl` (~5 MB)
3. **torchaudio**: `torchaudio-2.1.0+cu118-cp311-cp311-win_amd64.whl` (~5 MB)

**Note:** Adjust version numbers based on what's available. Look for:
- `cp311` = Python 3.11
- `cu118` = CUDA 11.8
- `win_amd64` = Windows 64-bit

### Step 2: Install Downloaded Wheels

```bash
cd Downloads  # Or wherever you saved the files
pip install torch-2.1.0+cu118-cp311-cp311-win_amd64.whl
pip install torchvision-0.16.0+cu118-cp311-cp311-win_amd64.whl
pip install torchaudio-2.1.0+cu118-cp311-cp311-win_amd64.whl
```

### Step 3: Install Other Requirements
```bash
cd "c:\Users\DELL\Documents\Rohith\Studies\Projects\Face_3D_Construction_Restoration\Face_3D_Reconstruction"
pip install -r requirements.txt
```

---

## ✅ Solution 4: Use Different Mirror (Slower but More Stable)

Try installing from PyPI instead of PyTorch's server:

```bash
# This installs a more compatible but potentially older version
pip install torch torchvision torchaudio
```

**Pros:** More reliable, rarely times out
**Cons:** Might install CPU version or older CUDA version

After installation, verify:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If it says `False`, you got CPU version. To fix:
```bash
pip uninstall torch torchvision torchaudio
# Then try Solution 1 or 2 again
```

---

## ✅ Solution 5: Increase Pip Timeout

Add this to your pip config:

```bash
# Create pip.ini (Windows)
mkdir "%APPDATA%\pip"
notepad "%APPDATA%\pip\pip.ini"
```

Add these lines:
```ini
[global]
timeout = 300
retries = 10
```

Save and close. Then try:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🔍 Verify Installation

After successful installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output:**
```
PyTorch: 2.1.0+cu118
CUDA Available: True
GPU: NVIDIA GeForce RTX 3050
```

---

## 💡 Tips for Success

### 1. **Use Wired Connection**
- WiFi can be unstable
- Wired ethernet is more reliable

### 2. **Close Other Applications**
- Stop downloads/streams
- Close browser tabs
- Free up bandwidth

### 3. **Try Different Times**
- Off-peak hours (night/early morning)
- PyTorch servers less busy

### 4. **Use Download Manager**
- Some browsers can resume interrupted downloads
- Use IDM or similar for wheel files

### 5. **Check Firewall/Antivirus**
- Might be blocking pip
- Temporarily disable and retry

---

## 🚨 Still Having Issues?

### Alternative: Install CPU Version First

If all else fails, install CPU version temporarily:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**You can:**
- Complete the project setup
- Download dataset
- Test code (will be slow)
- Later upgrade to CUDA when you have better connection

**To upgrade later:**
```bash
pip uninstall torch torchvision torchaudio
python setup_slow.py  # Try again with better connection
```

---

## 📞 Quick Commands Reference

```bash
# Recommended: Use slow connection installer
python setup_slow.py

# Or install manually one by one:
pip install --default-timeout=300 torch --index-url https://download.pytorch.org/whl/cu118
pip install --default-timeout=300 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install --default-timeout=300 torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"

# Install other requirements
pip install -r requirements.txt
```

---

## ✅ What to Do Now

**BEST OPTION:** Run the slow connection installer:
```bash
python setup_slow.py
```

This script:
- ✅ Handles timeouts automatically
- ✅ Retries with exponential backoff
- ✅ Shows real-time progress
- ✅ Falls back to CPU if CUDA fails
- ✅ Continues even if some packages fail
- ✅ Creates all necessary directories
- ✅ Verifies installation

**Expected time:** 15-30 minutes for full installation

---

## 🎉 After Successful Installation

Once PyTorch is installed:

```bash
# 1. Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# 2. Download dataset
python download_dataset.py

# 3. Quick test
python quick_train.py --epochs 5 --max-samples 1000

# 4. Start training
python quick_train.py --epochs 20 --max-samples 5000
```

Good luck! 🚀
