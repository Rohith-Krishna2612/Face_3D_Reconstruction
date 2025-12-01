# Face Restoration Web Application

## 🚀 Quick Start

### Prerequisites
```bash
# Python 3.8+ with dependencies installed
pip install -r requirements.txt

# Node.js 14+ with npm
cd frontend
npm install
```

### Running the Application

**Option 1: Automated Start (Windows)**
```bash
start_app.bat
```

**Option 2: Manual Start**

Terminal 1 - Backend:
```bash
cd backend
python main.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm start
```

## 📊 Application Flow

### Input
- Upload a face image (JPEG, PNG, BMP, WebP)
- Max size: 10MB
- Recommended: 512×512 resolution

### Processing
1. **Original Enhancement**: Model enhances the original image
2. **4 Degradations Applied**:
   - 🌫️ Gaussian Blur
   - 🔇 Gaussian Noise  
   - 📱 JPEG Compression
   - 🔍 Downsampling
3. **AI Restoration**: Model restores each degraded image

### Output
Total: **5 restored images**
- 1 enhanced original
- 4 restored degraded versions

## 🎨 Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Original Image Enhancement                                 │
│  ┌──────────────┐              ┌──────────────┐           │
│  │   Original   │      →       │   Enhanced   │           │
│  │    Input     │              │    Output    │           │
│  └──────────────┘              └──────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  4 Degradation Types & Restoration Results                  │
│                                                              │
│  1. Gaussian Blur                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Original │→ │ Degraded │→ │ Restored │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                              │
│  2. Gaussian Noise                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Original │→ │ Degraded │→ │ Restored │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                              │
│  3. JPEG Compression                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Original │→ │ Degraded │→ │ Restored │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                              │
│  4. Downsampling                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Original │→ │ Degraded │→ │ Restored │                │
│  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 API Endpoints

### Upload & Process Image
```
POST /upload-image/
Content-Type: multipart/form-data
Body: file (image file)

Response:
{
  "success": true,
  "original": "data:image/jpeg;base64,...",
  "original_restored": "data:image/jpeg;base64,...",
  "results": {
    "blur": {
      "degraded": "data:image/jpeg;base64,...",
      "restored": "data:image/jpeg;base64,..."
    },
    "gaussian_noise": { ... },
    "jpeg_compression": { ... },
    "downsampling": { ... }
  }
}
```

### Model Info
```
GET /model-info/
```

### Health Check
```
GET /health/
```

## 📝 Features

✅ **Drag & Drop Upload** - Easy image upload interface
✅ **Live Preview** - See your image before processing
✅ **4 Degradation Types** - Comprehensive testing
✅ **5 Restoration Outputs** - Original + 4 degraded versions
✅ **Click to Zoom** - Modal view for detailed inspection
✅ **Responsive Design** - Works on desktop and tablet
✅ **Error Handling** - User-friendly error messages
✅ **Progress Indicator** - Loading spinner during processing

## 🎯 Technical Details

### Backend (FastAPI)
- **Framework**: FastAPI with Uvicorn
- **Model**: CodeFormer (65M parameters)
- **Input Size**: 512×512 (resized automatically)
- **Processing Time**: ~2-5 seconds per degradation type
- **VRAM Usage**: ~2-3GB
- **CORS Enabled**: For local development

### Frontend (React)
- **Framework**: React 18
- **UI Library**: styled-components
- **File Upload**: react-dropzone
- **State Management**: useState hooks
- **API Calls**: Fetch API

### Degradation Parameters
```yaml
blur:
  kernel_sizes: [15, 21, 25]
  sigma_range: [0.1, 3.0]

gaussian_noise:
  noise_range: [0, 50]

jpeg_compression:
  quality_range: [10, 95]

downsampling:
  scale_factors: [2, 4, 8]
```

## 🐛 Troubleshooting

### Backend Issues

**Error: Model checkpoint not found**
```bash
# Train model first or use without checkpoint (will show warning)
python quick_train.py --epochs 5 --max-samples 1000
```

**Error: CUDA out of memory**
```bash
# Reduce resolution in config.yaml
dataset:
  resolution: 256  # Reduced from 512
```

### Frontend Issues

**Error: Cannot connect to backend**
```bash
# Check backend is running
curl http://localhost:8000/health/

# Check proxy in frontend/package.json
"proxy": "http://localhost:8000"
```

**Error: npm install fails**
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

## 📊 Performance

| Configuration | Processing Time | VRAM Usage |
|--------------|-----------------|------------|
| RTX 3050 (4GB) | ~3-4 sec/image | 2.5GB |
| RTX 3060 (12GB) | ~1-2 sec/image | 3GB |
| CPU Only | ~30-60 sec/image | N/A (16GB RAM) |

## 🎓 Academic Usage

This interface demonstrates:
- ✅ Real-time AI face restoration
- ✅ Multiple degradation types
- ✅ Before/after comparison
- ✅ Production-ready web application

Perfect for:
- Project demonstrations
- Academic presentations
- Research paper figures
- Interactive demos

## 📄 License

MIT License - see LICENSE file

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

---

**Need Help?** Open an issue on GitHub or check the main README.md
