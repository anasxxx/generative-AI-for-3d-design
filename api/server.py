#!/usr/bin/env python3
"""
API REST pour Fashion 2D-to-3D GAN
Endpoints pour génération et gestion du modèle
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import cv2
import io
import tempfile
import json
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from models.fashion_3d_gan import Fashion3DGAN, setup_gpu

# Configuration GPU
setup_gpu()

# Modèles de données API
class GenerationRequest(BaseModel):
    format: str = "obj"  # obj, ply, npy
    quality: str = "high"  # high, medium, fast

class GenerationResponse(BaseModel):
    success: bool
    message: str
    generation_id: str
    processing_time: float
    model_info: Optional[Dict] = None

class ModelStatus(BaseModel):
    model_loaded: bool
    model_path: str
    last_loaded: Optional[str] = None
    gpu_available: bool
    memory_usage: Optional[str] = None

class Fashion3DAPI:
    """API principale pour Fashion 2D-to-3D GAN"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Charger la configuration
        config_file = Path(__file__).parent.parent / config_path
        if config_file.exists():
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Configuration par défaut
            self.config = {
                'model_dir': './models/',
                'output_dir': './outputs/',
                'model_config': {
                    'input_resolution': [256, 256],
                    'voxel_resolution': 64,
                    'latent_dim': 512
                }
            }
        
        self.model_dir = Path(self.config['model_dir'])
        self.output_dir = Path(self.config['output_dir'])
        self.temp_dir = Path('./temp/')
        self.temp_dir.mkdir(exist_ok=True)
        
        # Variables d'état
        self.gan: Optional[Fashion3DGAN] = None
        self.model_loaded = False
        self.last_loaded = None
        self.model_path = None
        
        # Statistiques
        self.generation_count = 0
        self.total_processing_time = 0.0
        
        print("[OK] Fashion3D API initialized")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Charger le modèle GAN"""
        try:
            # Initialiser le GAN
            self.gan = Fashion3DGAN(
                img_shape=tuple(self.config['model_config']['input_resolution']) + (3,),
                voxel_size=self.config['model_config']['voxel_resolution'],
                latent_dim=self.config['model_config']['latent_dim']
            )
            
            # Load dataset info
            dataset_info_path = Path('./data/dataset_info.json')
            if dataset_info_path.exists():
                with open(dataset_info_path, 'r') as f:
                    self.dataset_info = json.load(f)
                print(f"[OK] Dataset info loaded: {self.dataset_info['total_samples']} samples")
            else:
                self.dataset_info = None
            
            self.model_loaded = True
            self.model_path = model_path or "demo_mode_with_filtered_mesh_dataset"
            self.last_loaded = datetime.now().isoformat()
            
            print(f"[OK] Model loaded with filtered_mesh dataset support")
            return True
            
        except Exception as e:
            print(f"[ERROR] Loading error: {e}")
            return False
    
    def preprocess_image(self, image_file: UploadFile) -> np.ndarray:
        """Préprocesser une image uploadée"""
        # Lire l'image
        image_bytes = image_file.file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Convertir BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionner à la taille attendue
        target_size = tuple(self.config['model_config']['input_resolution'])
        image = cv2.resize(image, target_size)
        
        # Normaliser [-1, 1] pour le GAN
        image = (image.astype(np.float32) / 127.5) - 1.0
        
        return image
    
    def generate_3d_model(self, image: np.ndarray, format: str = "obj") -> Dict:
        """Générer un modèle 3D à partir d'une image"""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = datetime.now()
        
        try:
            # Génération 3D
            voxels = self.gan.generate_3d(image)
            
            # Créer un ID unique pour cette génération
            generation_id = f"gen_{int(start_time.timestamp())}"
            
            # Calculer le temps de traitement
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Mettre à jour les statistiques
            self.generation_count += 1
            self.total_processing_time += processing_time
            
            return {
                'success': True,
                'generation_id': generation_id,
                'processing_time': processing_time,
                'voxel_occupancy': float(np.sum(voxels > 0.5) / voxels.size),
                'voxel_shape': voxels.shape
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")

# Initialiser l'API
api_instance = Fashion3DAPI()

# Créer l'application FastAPI
app = FastAPI(
    title="Fashion 2D-to-3D GAN API",
    description="API pour générer des modèles 3D à partir d'images 2D de mode",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Charger le modèle au démarrage"""
    print("[INFO] Loading model at startup...")
    api_instance.load_model()

@app.get("/", response_model=Dict)
async def root():
    """Endpoint racine avec informations sur l'API"""
    return {
        "message": "Fashion 2D-to-3D GAN API",
        "version": "1.0.0",
        "status": "running",
        "dataset": "Deep Fashion3D V2 - Filtered Mesh (1212 samples)",
        "endpoints": {
            "generate": "/generate - Generate 3D model from image",
            "status": "/model-status - Model status",
            "dataset": "/dataset-info - Dataset information",
            "stats": "/stats - Usage statistics",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.post("/generate", response_model=GenerationResponse)
async def generate_3d(
    file: UploadFile = File(...),
    format: str = "obj",
    quality: str = "high"
):
    """
    Générer un modèle 3D à partir d'une image 2D
    
    - **file**: Image 2D (JPEG, PNG, BMP)
    - **format**: Format de sortie (obj, ply, npy)
    - **quality**: Qualité de génération (high, medium, fast)
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Préprocesser l'image
        image = api_instance.preprocess_image(file)
        
        # Générer le modèle 3D
        result = api_instance.generate_3d_model(image, format)
        
        return GenerationResponse(
            success=result['success'],
            message="Generation successful (demo mode)",
            generation_id=result['generation_id'],
            processing_time=result['processing_time'],
            model_info={
                'voxel_occupancy': result['voxel_occupancy'],
                'voxel_shape': result['voxel_shape']
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status", response_model=ModelStatus)
async def get_model_status():
    """Obtenir l'état du modèle chargé"""
    # Vérifier l'utilisation GPU
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    memory_usage = None
    
    if gpu_available:
        try:
            # Obtenir info mémoire GPU (approximation)
            memory_usage = "GPU memory info available"
        except:
            memory_usage = "Not available"
    
    return ModelStatus(
        model_loaded=api_instance.model_loaded,
        model_path=api_instance.model_path or "Not loaded",
        last_loaded=api_instance.last_loaded,
        gpu_available=gpu_available,
        memory_usage=memory_usage
    )

@app.get("/dataset-info")
async def get_dataset_info():
    """Get information about the loaded dataset"""
    if hasattr(api_instance, 'dataset_info') and api_instance.dataset_info:
        return api_instance.dataset_info
    else:
        return {"message": "No dataset information available", "dataset_loaded": False}

@app.get("/stats")
async def get_stats():
    """Statistiques d'utilisation de l'API"""
    avg_processing_time = (
        api_instance.total_processing_time / api_instance.generation_count
        if api_instance.generation_count > 0 else 0
    )
    
    return {
        'total_generations': api_instance.generation_count,
        'total_processing_time': api_instance.total_processing_time,
        'average_processing_time': avg_processing_time,
        'model_loaded': api_instance.model_loaded,
        'uptime': datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("[INFO] Starting Fashion 3D API server...")
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # No reload in production
        workers=1      # Single worker for TensorFlow model
    )
