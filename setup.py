"""
Setup script to install dependencies and prepare environment.
"""

import subprocess
import sys
import os


def run_command(command, description="", retries=3, timeout=600):
    """Run a command and handle errors with retry logic."""
    print(f"Running: {description}")
    print(f"Command: {command}")
    
    for attempt in range(retries):
        if attempt > 0:
            print(f"Retry attempt {attempt + 1}/{retries}...")
        
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                check=True, 
                capture_output=True, 
                text=True,
                timeout=timeout
            )
            print(f"✓ Success: {description}")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.TimeoutExpired:
            print(f"⏱️  Timeout on attempt {attempt + 1}/{retries}")
            if attempt == retries - 1:
                print(f"✗ Failed after {retries} attempts: {description}")
                return False
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {description}")
            print(f"Error: {e.stderr}")
            return False
    
    return False


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✓ Python version is compatible")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        return False


def setup_environment():
    """Setup the development environment."""
    print("🚀 Setting up Face 3D Reconstruction Environment")
    print("=" * 60)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if not check_python_version():
        return False
    
    # Install PyTorch
    print("\n2. Installing PyTorch...")
    
    # Detect CUDA availability
    print("Checking for CUDA...")
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, shell=True)
        has_cuda = result.returncode == 0
    except:
        has_cuda = False
    
    if has_cuda:
        print("✅ NVIDIA GPU detected! Installing CUDA version of PyTorch...")
        print("\nNote: PyTorch download is ~2GB and may take several minutes.")
        print("If it times out, the script will retry automatically.\n")
        
        # CUDA 11.8 is widely compatible with RTX 3050
        # Increase timeout and pip timeout setting
        pytorch_command = "pip install --default-timeout=300 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        install_type = "Installing PyTorch (CUDA 11.8)"
        
        # Try with extended timeout (10 minutes per attempt, 3 attempts = 30 min max)
        if not run_command(pytorch_command, install_type, retries=3, timeout=600):
            print("\n⚠️  PyTorch installation timed out or failed.")
            print("\nAlternative installation methods:")
            print("\n1. Manual installation (RECOMMENDED):")
            print("   pip install --default-timeout=300 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            print("\n2. Or try downloading pre-built wheels:")
            print("   Visit: https://download.pytorch.org/whl/torch_stable.html")
            print("   Download: torch-*-cu118-*.whl, torchvision-*-cu118-*.whl, torchaudio-*.whl")
            print("   Install: pip install <downloaded_wheel_file>")
            print("\n3. Use slower but more reliable mirror:")
            print("   pip install torch torchvision torchaudio")
            print("   (This installs CPU version first, then upgrade to CUDA)")
            
            retry = input("\nWould you like to try a simpler installation? (y/n): ").lower() == 'y'
            if retry:
                print("\nTrying alternative method (install one by one)...")
                print("This is slower but more reliable for slow connections.")
                
                # Install torch first
                if run_command("pip install --default-timeout=300 torch --index-url https://download.pytorch.org/whl/cu118", 
                             "Installing torch only", retries=3, timeout=600):
                    # Then torchvision
                    if run_command("pip install --default-timeout=300 torchvision --index-url https://download.pytorch.org/whl/cu118",
                                 "Installing torchvision", retries=2, timeout=600):
                        # Finally torchaudio
                        run_command("pip install --default-timeout=300 torchaudio --index-url https://download.pytorch.org/whl/cu118",
                                  "Installing torchaudio", retries=2, timeout=600)
                        print("✅ PyTorch installation completed via alternative method!")
                    else:
                        print("❌ Installation failed. Please try manual installation.")
                        return False
                else:
                    print("❌ Installation failed. Please try manual installation.")
                    return False
            else:
                return False
    else:
        print("⚠️  No NVIDIA GPU detected. Installing CPU version...")
        pytorch_command = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        install_type = "Installing PyTorch (CPU)"
        
        if not run_command(pytorch_command, install_type, retries=3, timeout=600):
            return False
    
    # Install other requirements
    print("\n3. Installing other dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    # Create necessary directories
    print("\n4. Creating project directories...")
    directories = [
        "data/input",
        "data/enhanced", 
        "data/output_3d",
        "checkpoints",
        "logs",
        "output",
        "models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\n5. Environment setup completed!")
    print("Next steps:")
    print("1. Verify CUDA: python -c \"import torch; print(torch.cuda.is_available())\"")
    print("2. Download dataset manually from Kaggle:")
    print("   https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq")
    print("   Extract to data\\ffhq\\")
    print("3. Quick test: python quick_train.py --epochs 5 --max-samples 1000")
    print("4. Start training: python quick_train.py --epochs 20 --max-samples 5000")
    
    return True


def setup_kaggle():
    """Setup Kaggle API credentials."""
    print("\n📊 Setting up Kaggle API")
    print("To download FFHQ dataset, you need Kaggle API credentials.")
    print("\nSteps:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'") 
    print("3. Download kaggle.json")
    print("4. Place it in:")
    
    if os.name == 'nt':  # Windows
        kaggle_dir = os.path.expanduser("~/.kaggle")
        print(f"   Windows: {kaggle_dir}")
    else:  # Unix/Linux/Mac
        kaggle_dir = os.path.expanduser("~/.kaggle")
        print(f"   Unix/Linux/Mac: {kaggle_dir}")
    
    print("5. Run: chmod 600 ~/.kaggle/kaggle.json (Unix/Linux/Mac only)")
    
    # Create kaggle directory
    os.makedirs(kaggle_dir, exist_ok=True)
    
    kaggle_path = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_path):
        print(f"✓ Kaggle credentials found at: {kaggle_path}")
    else:
        print(f"! Please place kaggle.json at: {kaggle_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup Face 3D Reconstruction environment')
    parser.add_argument('--kaggle-setup', action='store_true',
                       help='Show Kaggle API setup instructions')
    
    args = parser.parse_args()
    
    if args.kaggle_setup:
        setup_kaggle()
    else:
        success = setup_environment()
        
        if success:
            print("\n" + "🎉 Setup completed successfully!" + "\n")
            
            setup_kaggle_now = input("Would you like to see Kaggle setup instructions? (y/n): ").lower() == 'y'
            if setup_kaggle_now:
                setup_kaggle()
        else:
            print("\n" + "❌ Setup failed. Please check the errors above." + "\n")