"""
FastAPI backend for face restoration and 3D reconstruction.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import torch
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys
import uuid
from typing import List, Dict, Any
import yaml
from pathlib import Path
import base64

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.codeformer import create_codeformer_model
from src.degradations import create_degradation_pipeline
from src.utils import (
    load_image, save_image, tensor_to_numpy, numpy_to_tensor,
    get_device, load_checkpoint
)

# Initialize FastAPI app
app = FastAPI(
    title="Face 3D Reconstruction API",
    description="API for face restoration and 3D reconstruction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
degradation_pipeline = None
device = None
config = None


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_model():
    """Initialize the CodeFormer model."""
    global model, degradation_pipeline, device, config
    
    # Load configuration
    config = load_config()
    
    # Get device
    device = get_device()
    
    # Create model
    model = create_codeformer_model(config).to(device)
    
    # Load trained weights if available
    checkpoint_path = os.path.join("checkpoints", "codeformer", "best_checkpoint.pth")
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded model from: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
    
    # Set to evaluation mode
    model.eval()
    
    # Create degradation pipeline
    degradation_pipeline = create_degradation_pipeline(config)
    
    print("Model initialized successfully!")


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    initialize_model()


def process_uploaded_image(file_content: bytes) -> np.ndarray:
    """Process uploaded image file."""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        return image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


def apply_degradations(image: np.ndarray) -> Dict[str, np.ndarray]:
    """Apply all degradation types to image."""
    return degradation_pipeline.apply_multiple_degradations(image)


def restore_image(degraded_image: np.ndarray, restoration_strength: float = 0.5) -> np.ndarray:
    """Restore image using CodeFormer model."""
    try:
        # Resize to model input size
        target_size = (512, 512)
        degraded_resized = cv2.resize(degraded_image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        input_tensor = numpy_to_tensor(degraded_resized, device)
        
        # Normalize to [-1, 1]
        input_tensor = input_tensor * 2.0 - 1.0
        
        # Run inference
        with torch.no_grad():
            restored_tensor, _ = model(input_tensor, w=restoration_strength)
        
        # Convert back to numpy
        restored_np = tensor_to_numpy(restored_tensor)
        
        return restored_np
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during restoration: {str(e)}")


def numpy_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64 string."""
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    
    # Encode to base64
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image and get degradations + restorations."""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process image
        original_image = process_uploaded_image(file_content)
        
        # Apply degradations
        degraded_images = apply_degradations(original_image)
        
        # Restore each degraded image
        restoration_results = {}
        
        for degradation_type, degraded_image in degraded_images.items():
            if degradation_type == 'original':
                continue
                
            # Restore the degraded image
            restored_image = restore_image(degraded_image)
            
            restoration_results[degradation_type] = {
                'degraded': numpy_to_base64(degraded_image),
                'restored': numpy_to_base64(restored_image)
            }
        
        return {
            'success': True,
            'original': numpy_to_base64(original_image),
            'results': restoration_results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/restore-custom/")
async def restore_custom_image(
    file: UploadFile = File(...), 
    strength: float = 0.5
):
    """Restore a custom uploaded image with specified strength."""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Validate strength parameter
    if not 0.0 <= strength <= 1.0:
        raise HTTPException(status_code=400, detail="Strength must be between 0.0 and 1.0")
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Process image
        input_image = process_uploaded_image(file_content)
        
        # Restore image
        restored_image = restore_image(input_image, restoration_strength=strength)
        
        return {
            'success': True,
            'original': numpy_to_base64(input_image),
            'restored': numpy_to_base64(restored_image),
            'strength_used': strength
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info/")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        return {"model_loaded": False}
    
    try:
        param_count = sum(p.numel() for p in model.parameters())
        
        return {
            "model_loaded": True,
            "model_type": "CodeFormer",
            "device": str(device),
            "parameter_count": param_count,
            "input_size": config.get('model', {}).get('input_size', [512, 512])
        }
    except Exception as e:
        return {"model_loaded": True, "error": str(e)}


@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device) if device else "unknown"
    }


@app.get("/degradation-types/")
async def get_degradation_types():
    """Get available degradation types."""
    return {
        "degradation_types": [
            {
                "name": "blur",
                "description": "Gaussian blur with various kernel sizes"
            },
            {
                "name": "gaussian_noise", 
                "description": "Additive Gaussian noise"
            },
            {
                "name": "jpeg_compression",
                "description": "JPEG compression artifacts"
            },
            {
                "name": "downsampling",
                "description": "Downsampling and upsampling degradation"
            }
        ]
    }


# Mount static files for serving frontend (if needed)
if os.path.exists("frontend/build"):
    app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")
    
    @app.get("/")
    async def serve_frontend():
        return FileResponse("frontend/build/index.html")


if __name__ == "__main__":
    import uvicorn
    
    # Load config for port
    try:
        config = load_config()
        port = config.get('api', {}).get('port', 8000)
        host = config.get('api', {}).get('host', '0.0.0.0')
    except:
        port = 8000
        host = '0.0.0.0'
    
    uvicorn.run(
        "main:app", 
        host=host, 
        port=port, 
        reload=True,
        log_level="info"
    )