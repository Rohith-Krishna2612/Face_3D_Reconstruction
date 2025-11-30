"""
Alternative setup script for slow or unstable internet connections.
Installs packages one by one with better error handling.
"""

import subprocess
import sys
import os
import time


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_pip_install(package, description="", retries=3):
    """Install a single package with retry logic."""
    print(f"\n📦 {description or f'Installing {package}'}")
    
    for attempt in range(retries):
        if attempt > 0:
            wait_time = 5 * attempt
            print(f"⏳ Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
            print(f"🔄 Retry attempt {attempt + 1}/{retries}...")
        
        try:
            # Use --no-cache-dir to avoid cache issues
            # Use --default-timeout for longer timeout
            cmd = [
                sys.executable, "-m", "pip", "install",
                "--default-timeout=300",
                "--retries", "5",
                "--no-cache-dir",
                package
            ]
            
            print(f"Running: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            print(f"✅ Successfully installed: {package}")
            return True
            
        except subprocess.TimeoutExpired:
            print(f"⏱️  Installation timed out (attempt {attempt + 1}/{retries})")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed (attempt {attempt + 1}/{retries})")
            print(f"Error code: {e.returncode}")
            
        except KeyboardInterrupt:
            print("\n⚠️  Installation cancelled by user")
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    print(f"💥 Failed to install {package} after {retries} attempts")
    return False


def check_cuda():
    """Check if CUDA is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            # Extract GPU name
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    print(f"   GPU: {line.strip()}")
                    break
            return True
    except:
        pass
    
    print("⚠️  No NVIDIA GPU detected")
    return False


def install_pytorch_cuda():
    """Install PyTorch with CUDA support step by step."""
    print_header("Installing PyTorch with CUDA 11.8")
    
    print("\n💡 Tips for successful installation:")
    print("   - Keep this window open")
    print("   - Don't interrupt the download")
    print("   - If it fails, the script will auto-retry")
    print("   - Total download: ~2 GB")
    print("   - Expected time: 5-15 minutes")
    
    packages = [
        ("torch", "https://download.pytorch.org/whl/cu118"),
        ("torchvision", "https://download.pytorch.org/whl/cu118"),
        ("torchaudio", "https://download.pytorch.org/whl/cu118"),
    ]
    
    for package_name, index_url in packages:
        package = f"{package_name} --index-url {index_url}"
        if not run_pip_install(package, f"Installing {package_name} (CUDA 11.8)"):
            print(f"\n❌ Failed to install {package_name}")
            print("\n🔧 Troubleshooting options:")
            print("1. Check your internet connection")
            print("2. Try using a VPN or different network")
            print("3. Download manually from: https://download.pytorch.org/whl/torch_stable.html")
            print(f"4. Or install CPU version: pip install {package_name}")
            return False
    
    return True


def install_pytorch_cpu():
    """Install PyTorch CPU version."""
    print_header("Installing PyTorch (CPU version)")
    
    packages = [
        ("torch", "https://download.pytorch.org/whl/cpu"),
        ("torchvision", "https://download.pytorch.org/whl/cpu"),
        ("torchaudio", "https://download.pytorch.org/whl/cpu"),
    ]
    
    for package_name, index_url in packages:
        package = f"{package_name} --index-url {index_url}"
        if not run_pip_install(package, f"Installing {package_name} (CPU)"):
            return False
    
    return True


def install_requirements():
    """Install other requirements from requirements.txt."""
    print_header("Installing Other Dependencies")
    
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False
    
    # Read requirements and install one by one
    with open("requirements.txt", 'r') as f:
        lines = f.readlines()
    
    packages = []
    for line in lines:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith('#'):
            packages.append(line)
    
    print(f"\n📋 Found {len(packages)} packages to install")
    print("Installing one by one for better reliability...\n")
    
    failed_packages = []
    
    for i, package in enumerate(packages, 1):
        # Skip torch packages as they're already installed
        if any(x in package.lower() for x in ['torch', 'torchvision', 'torchaudio']):
            print(f"[{i}/{len(packages)}] ⏭️  Skipping {package} (already installed)")
            continue
        
        print(f"[{i}/{len(packages)}] Installing {package}")
        
        if not run_pip_install(package, f"Installing {package}", retries=2):
            failed_packages.append(package)
            print(f"⚠️  Failed to install {package}, continuing with others...")
    
    if failed_packages:
        print("\n⚠️  Some packages failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\nYou can try installing them manually later:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
        return False
    
    return True


def create_directories():
    """Create necessary project directories."""
    print_header("Creating Project Directories")
    
    directories = [
        "data/input",
        "data/enhanced",
        "data/output_3d",
        "data/ffhq",
        "checkpoints",
        "checkpoints/codeformer",
        "logs",
        "logs/codeformer",
        "output",
        "output/samples",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True


def verify_installation():
    """Verify PyTorch installation."""
    print_header("Verifying Installation")
    
    try:
        print("\n🔍 Checking PyTorch installation...")
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        print("\n🔍 Checking CUDA availability...")
        if torch.cuda.is_available():
            print(f"✅ CUDA is available!")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  CUDA is not available (CPU mode)")
            print("   This is OK, but training will be much slower")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import PyTorch: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False


def main():
    """Main installation process."""
    print("\n" + "🚀 " * 20)
    print("  CodeFormer Setup - Slow Connection Mode")
    print("🚀 " * 20)
    
    print("\n📌 This installer is optimized for:")
    print("   - Slow internet connections")
    print("   - Unstable networks")
    print("   - Timeout issues")
    print("\n⏱️  Expected time: 15-30 minutes")
    print("💾 Total download: ~3 GB")
    
    input("\nPress Enter to start installation...")
    
    # Step 1: Check CUDA
    print_header("Step 1/5: Checking GPU")
    has_cuda = check_cuda()
    
    # Step 2: Upgrade pip
    print_header("Step 2/5: Upgrading pip")
    run_pip_install("--upgrade pip setuptools wheel", "Upgrading pip", retries=2)
    
    # Step 3: Install PyTorch
    print_header("Step 3/5: Installing PyTorch")
    if has_cuda:
        if not install_pytorch_cuda():
            print("\n⚠️  PyTorch CUDA installation failed")
            fallback = input("Install CPU version instead? (y/n): ").lower() == 'y'
            if fallback:
                if not install_pytorch_cpu():
                    print("❌ Setup failed")
                    return False
            else:
                print("❌ Setup cancelled")
                return False
    else:
        if not install_pytorch_cpu():
            print("❌ Setup failed")
            return False
    
    # Step 4: Install other requirements
    print_header("Step 4/5: Installing Other Dependencies")
    install_requirements()  # Don't fail if some packages fail
    
    # Step 5: Create directories
    print_header("Step 5/5: Creating Directories")
    create_directories()
    
    # Verify installation
    if verify_installation():
        print_header("🎉 Installation Completed Successfully!")
        
        print("\n✅ Next steps:")
        print("   1. Download dataset manually from Kaggle:")
        print("      https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
        print("      Extract to data\\ffhq\\")
        print("   2. Quick test: python quick_train.py --epochs 5 --max-samples 1000")
        print("   3. Start training: python quick_train.py --epochs 20 --max-samples 5000")
        
        print("\n📚 Documentation:")
        print("   - Quick start: QUICK_START_MANUAL.md")
        print("   - RTX 3050 guide: RTX_3050_OPTIMIZATION_GUIDE.md")
        
        return True
    else:
        print_header("⚠️ Installation completed with warnings")
        print("Please check the messages above and try to fix any issues.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
