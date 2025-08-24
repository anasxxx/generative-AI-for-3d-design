#!/usr/bin/env python3
"""
Convert raw fashion data to numpy format for training
"""

import os
import sys
import numpy as np
import json
from pathlib import Path
from PIL import Image
import trimesh
from tqdm import tqdm
import argparse

def load_mesh_as_voxel(mesh_path, voxel_size=64):
    """Convert mesh to voxel grid"""
    try:
        # Load mesh
        mesh = trimesh.load(str(mesh_path))
        
        # Normalize mesh to fit in [-1, 1] cube
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.vertices /= np.max(np.abs(mesh.vertices)) * 1.1
        
        # Create voxel grid
        voxel_grid = np.zeros((voxel_size, voxel_size, voxel_size))
        
        # Sample points from mesh surface
        points, _ = mesh.sample(10000)
        
        # Convert to voxel coordinates
        voxel_coords = ((points + 1) * (voxel_size - 1) / 2).astype(int)
        voxel_coords = np.clip(voxel_coords, 0, voxel_size - 1)
        
        # Fill voxel grid
        for x, y, z in voxel_coords:
            voxel_grid[x, y, z] = 1.0
        
        return voxel_grid
    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        # Return empty voxel grid
        return np.zeros((voxel_size, voxel_size, voxel_size))

def convert_dataset_to_numpy():
    """Convert raw dataset to numpy format"""
    print("🚀 Converting Fashion Dataset to Numpy Format")
    print("=" * 50)
    
    # Load dataset info
    dataset_info_path = Path('./data/dataset_info.json')
    if not dataset_info_path.exists():
        print("❌ Dataset info not found. Run preprocessing first.")
        return
    
    with open(dataset_info_path, 'r') as f:
        dataset_info = json.load(f)
    
    # Create output directories
    train_dir = Path('./data/train')
    val_dir = Path('./data/val')
    
    for dir_path in [train_dir, val_dir]:
        (dir_path / 'images').mkdir(parents=True, exist_ok=True)
        (dir_path / 'voxels').mkdir(parents=True, exist_ok=True)
    
    # Load pairs from dataset info and create train/val split
    all_pairs = dataset_info['train_samples_info']
    
    # Create train/val split (80/20)
    split_idx = int(0.8 * len(all_pairs))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"📊 Total samples: {len(all_pairs)}")
    print(f"📊 Training samples: {len(train_pairs)}")
    print(f"📊 Validation samples: {len(val_pairs)}")
    
    print(f"📊 Converting {len(train_pairs)} training samples...")
    
    # Convert training data
    for i, pair_info in enumerate(tqdm(train_pairs, desc="Training data")):
        sample_id = pair_info['id']
        
        # Construct paths
        sample_dir = Path(f"C:/Users/mahmo/OneDrive/Documents/filtered_mesh/{sample_id}")
        image_path = sample_dir / f"{sample_id}_tex.png"
        mesh_path = sample_dir / "model_cleaned.obj"
        
        if image_path.exists() and mesh_path.exists():
            try:
                # Load and process image
                img = Image.open(image_path).convert('RGB')
                img = img.resize((256, 256))
                img_array = np.array(img).astype(np.float32) / 255.0 * 2 - 1  # [-1, 1]
                
                # Load and process mesh
                voxel = load_mesh_as_voxel(mesh_path)
                
                # Save as numpy files
                np.save(train_dir / 'images' / f"{sample_id}.npy", img_array)
                np.save(train_dir / 'voxels' / f"{sample_id}.npy", voxel)
                
            except Exception as e:
                print(f"❌ Error processing {sample_id}: {e}")
                continue
    
    print(f"📊 Converting {len(val_pairs)} validation samples...")
    
    # Convert validation data
    for i, pair_info in enumerate(tqdm(val_pairs, desc="Validation data")):
        sample_id = pair_info['id']
        
        # Construct paths
        sample_dir = Path(f"C:/Users/mahmo/OneDrive/Documents/filtered_mesh/{sample_id}")
        image_path = sample_dir / f"{sample_id}_tex.png"
        mesh_path = sample_dir / "model_cleaned.obj"
        
        if image_path.exists() and mesh_path.exists():
            try:
                # Load and process image
                img = Image.open(image_path).convert('RGB')
                img = img.resize((256, 256))
                img_array = np.array(img).astype(np.float32) / 255.0 * 2 - 1  # [-1, 1]
                
                # Load and process mesh
                voxel = load_mesh_as_voxel(mesh_path)
                
                # Save as numpy files
                np.save(val_dir / 'images' / f"{sample_id}.npy", img_array)
                np.save(val_dir / 'voxels' / f"{sample_id}.npy", voxel)
                
            except Exception as e:
                print(f"❌ Error processing {sample_id}: {e}")
                continue
    
    print("✅ Conversion completed!")
    print(f"📁 Training data: {len(list((train_dir / 'images').glob('*.npy')))} samples")
    print(f"📁 Validation data: {len(list((val_dir / 'images').glob('*.npy')))} samples")

if __name__ == "__main__":
    convert_dataset_to_numpy()
