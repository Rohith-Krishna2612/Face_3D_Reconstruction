# Face 3D Reconstruction 🎭

A comprehensive AI-powered face restoration and 3D reconstruction pipeline using CodeFormer architecture with FFHQ dataset training.

## ✨ Features

- **Face Restoration**: Advanced AI model for restoring degraded face images
- **Multiple Degradations**: Handles blur, noise, JPEG compression, and downsampling
- **Side-by-Side Comparison**: Interactive web interface showing before/after results
- **Training Pipeline**: Complete training setup with FFHQ dataset support
- **Real-time Processing**: FastAPI backend with React frontend

## 🏗️ Architecture

### Phase 1: Face Enhancement & Restoration
- **Model**: CodeFormer (VQGAN + Transformer)
- **Dataset**: FFHQ (Flickr-Faces-HQ)
- **Degradations**: 4 types with configurable parameters
- **Training**: Full pipeline with checkpointing and monitoring

### Phase 2: Web Application
- **Backend**: FastAPI with model inference
- **Frontend**: React with modern UI components
- **Features**: Image upload, real-time processing, comparison grid

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and setup
git clone <your-repo>
cd Face_3D_Reconstruction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run setup script
python setup.py
```

### 2. Install Dependencies
```bash
# Install PyTorch (choose your version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

### 3. Dataset Preparation
```bash
# Setup Kaggle API credentials first
# Then run dataset setup
python train.py --setup-only
```

### 4. Training (Optional)
```bash
# Start training CodeFormer
python train.py

# Monitor progress
tensorboard --logdir logs/
```

### 5. Run Application
```bash
# Windows
start_dev.bat

# Linux/Mac
bash start_dev.sh

# Or manually:
# Terminal 1: python backend/main.py
# Terminal 2: cd frontend && npm start
```

## 📁 Project Structure

```
Face_3D_Reconstruction/
├── src/
│   ├── models/
│   │   ├── codeformer.py      # CodeFormer architecture
│   │   ├── discriminator.py   # GAN discriminator
│   │   └── losses.py          # Loss functions
│   ├── degradations/
│   │   └── degradation_pipeline.py  # Image degradations
│   ├── training/
│   │   └── train_codeformer.py      # Training script
│   └── utils/
│       ├── image_utils.py     # Image processing utilities
│       └── dataset_utils.py   # Dataset handling
├── backend/
│   └── main.py               # FastAPI server
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   └── App.js           # Main app
│   └── package.json         # Dependencies
├── data/                    # Dataset directory
├── models/                  # Trained models
├── checkpoints/            # Training checkpoints
└── config.yaml             # Configuration
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Degradation settings
- Dataset paths

## 🎯 Usage

### Training Your Own Model
1. **Prepare Dataset**: Download FFHQ or use your own face dataset
2. **Configure**: Update `config.yaml` with your settings
3. **Train**: Run `python train.py`
4. **Monitor**: Use TensorBoard or Weights & Biases

### Using Pre-trained Model
1. **Download**: Place pre-trained weights in `checkpoints/`
2. **Configure**: Update model paths in config
3. **Run**: Start the web application

### Web Interface
1. **Upload**: Drag & drop or select face image
2. **Process**: AI applies degradations and restorations
3. **Compare**: View side-by-side results
4. **Download**: Save restored images

## 📊 Degradation Types

| Type | Description | Parameters |
|------|-------------|------------|
| 🌫️ Blur | Gaussian blur | Kernel size, sigma |
| 🔇 Noise | Gaussian noise | Noise level (0-50) |
| 📱 JPEG | Compression artifacts | Quality (10-95) |
| 🔍 Downsampling | Resolution reduction | Scale factors (2x, 4x, 8x) |

## 🛠️ Development

### Backend Development
```bash
cd backend
python main.py
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend Development
```bash
cd frontend
npm install
npm start
# React app at http://localhost:3000
```

### Adding New Features
1. **New Degradation**: Add to `degradation_pipeline.py`
2. **New Model**: Implement in `src/models/`
3. **UI Changes**: Modify React components

## 📈 Performance

- **Training**: ~24 hours on RTX 3090 for 100 epochs
- **Inference**: ~0.5-2 seconds per 512x512 image
- **Memory**: ~8GB GPU memory for training

## 🐛 Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size in config
2. **Dataset Not Found**: Check paths in `config.yaml`
3. **Import Errors**: Ensure virtual environment is activated
4. **Slow Inference**: Use GPU version of PyTorch

### Getting Help
- Check logs in `logs/` directory
- Enable debug mode in config
- Review error messages in terminal

## 📝 TODO / Future Work

- [ ] Add more degradation types
- [ ] Implement 3D face reconstruction phase
- [ ] Add batch processing support
- [ ] Mobile-friendly UI
- [ ] Docker containerization
- [ ] Model quantization for faster inference

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- CodeFormer paper and implementation
- FFHQ dataset by NVIDIA
- PyTorch and React communities
- FastAPI framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Happy Face Restoration! 🎭✨**
