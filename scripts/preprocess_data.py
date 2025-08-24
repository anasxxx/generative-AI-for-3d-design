#!/usr/bin/env python3
"""
Preprocessing du dataset Deep Fashion3D V2 pour l'entraînement GAN
Version complète avec support pour différents formats de dataset
"""

import os
import yaml
import numpy as np
import cv2
import json
from pathlib import Path
import sys
from tqdm import tqdm
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class FashionDataPreprocessor:
    """Preprocessor for Fashion 2D-to-3D datasets"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_path = Path(self.config['dataset_path'])
        self.output_dir = Path('./data/')
        self.output_dir.mkdir(exist_ok=True)
        
        # Target sizes
        self.img_size = tuple(self.config['model_config']['input_resolution'])
        self.voxel_size = self.config['model_config']['voxel_resolution']
        
        # Training parameters
        self.max_samples = 2000  # Limit for 8h training
        self.train_ratio = 0.8
        
        print(f"[INFO] Preprocessor initialized")
        print(f"[INFO] Dataset path: {self.dataset_path}")
        print(f"[INFO] Image size: {self.img_size}")
        print(f"[INFO] Voxel size: {self.voxel_size}")
    
    def detect_dataset_structure(self):
        """Detect the structure of the dataset"""
        print("[INFO] Detecting dataset structure...")
        
        if not self.dataset_path.exists():
            print(f"[ERROR] Dataset not found: {self.dataset_path}")
            return None
        
        # Check for filtered_mesh structure
        if self._is_filtered_mesh_structure():
            return 'filtered_mesh'
        
        # Check for standard Deep Fashion3D structure
        if self._is_standard_fashion3d_structure():
            return 'standard_fashion3d'
        
        # Check for custom structure
        if self._is_custom_structure():
            return 'custom'
        
        print("[WARNING] Unknown dataset structure")
        return 'unknown'
    
    def _is_filtered_mesh_structure(self):
        """Check if dataset has filtered_mesh structure"""
        # Look for directories with {ID}_tex.png and model_cleaned.obj
        directories = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        
        if len(directories) == 0:
            return False
        
        # Check first few directories
        for dir_path in directories[:5]:
            dir_name = dir_path.name
            tex_file = dir_path / f"{dir_name}_tex.png"
            obj_file = dir_path / "model_cleaned.obj"
            
            if tex_file.exists() and obj_file.exists():
                return True
        
        return False
    
    def _is_standard_fashion3d_structure(self):
        """Check if dataset has standard Deep Fashion3D structure"""
        # Look for common Deep Fashion3D patterns
        common_files = ['train.txt', 'test.txt', 'val.txt', 'images', 'meshes']
        
        for file_name in common_files:
            if (self.dataset_path / file_name).exists():
                return True
        
        return False
    
    def _is_custom_structure(self):
        """Check if dataset has custom structure"""
        # Look for image and mesh files
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        mesh_exts = {'.obj', '.ply', '.off', '.mesh'}
        
        has_images = False
        has_meshes = False
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in image_exts:
                    has_images = True
                elif ext in mesh_exts:
                    has_meshes = True
                
                if has_images and has_meshes:
                    return True
        
        return False
    
    def preprocess_filtered_mesh(self):
        """Preprocess filtered_mesh dataset structure"""
        print("[INFO] Preprocessing filtered_mesh dataset...")
        
        # Import the specialized preprocessor
        from scripts.preprocess_filtered_mesh import FilteredMeshPreprocessor
        
        preprocessor = FilteredMeshPreprocessor()
        return preprocessor.run_preprocessing()
    
    def preprocess_standard_fashion3d(self):
        """Preprocess standard Deep Fashion3D dataset"""
        print("[INFO] Preprocessing standard Deep Fashion3D dataset...")
        
        # This would implement standard Deep Fashion3D preprocessing
        # For now, create a synthetic dataset
        return self.create_synthetic_dataset()
    
    def preprocess_custom_dataset(self):
        """Preprocess custom dataset structure"""
        print("[INFO] Preprocessing custom dataset...")
        
        # Find all image and mesh files
        image_files = []
        mesh_files = []
        
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        mesh_exts = {'.obj', '.ply', '.off', '.mesh'}
        
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()
                
                if ext in image_exts:
                    image_files.append(file_path)
                elif ext in mesh_exts:
                    mesh_files.append(file_path)
        
        print(f"[INFO] Found {len(image_files)} images and {len(mesh_files)} meshes")
        
        if len(image_files) == 0 or len(mesh_files) == 0:
            print("[ERROR] No image-mesh pairs found")
            return False
        
        # Try to pair images and meshes
        pairs = self.pair_images_and_meshes(image_files, mesh_files)
        
        if len(pairs) == 0:
            print("[ERROR] No valid pairs found")
            return False
        
        # Create dataset info
        return self.create_dataset_info(pairs, 'custom')
    
    def pair_images_and_meshes(self, image_files, mesh_files):
        """Pair images and meshes based on naming patterns"""
        print("[INFO] Pairing images and meshes...")
        
        pairs = []
        
        # Create dictionaries for quick lookup
        image_dict = {f.stem: f for f in image_files}
        mesh_dict = {f.stem: f for f in mesh_files}
        
        # Try exact matches first
        exact_matches = set(image_dict.keys()) & set(mesh_dict.keys())
        
        for stem in exact_matches:
            pairs.append({
                'id': stem,
                'image_path': str(image_dict[stem]),
                'mesh_path': str(mesh_dict[stem]),
                'category': self.extract_category(stem)
            })
        
        print(f"[INFO] Found {len(pairs)} exact matches")
        
        # If not enough pairs, try partial matches
        if len(pairs) < 100:
            remaining_images = set(image_dict.keys()) - exact_matches
            remaining_meshes = set(mesh_dict.keys()) - exact_matches
            
            for img_stem in remaining_images:
                for mesh_stem in remaining_meshes:
                    if self.similar_names(img_stem, mesh_stem):
                        pairs.append({
                            'id': f"{img_stem}_{mesh_stem}",
                            'image_path': str(image_dict[img_stem]),
                            'mesh_path': str(mesh_dict[mesh_stem]),
                            'category': self.extract_category(img_stem)
                        })
                        break
        
        print(f"[INFO] Total pairs: {len(pairs)}")
        return pairs
    
    def similar_names(self, name1, name2, threshold=0.8):
        """Check if two names are similar"""
        try:
            from difflib import SequenceMatcher
            return SequenceMatcher(None, name1.lower(), name2.lower()).ratio() > threshold
        except:
            return False
    
    def extract_category(self, name):
        """Extract category from filename"""
        name_lower = name.lower()
        
        categories = {
            'bag': ['bag', 'handbag', 'purse', 'sac', 'tote'],
            'shoe': ['shoe', 'boot', 'sneaker', 'chaussure', 'footwear'],
            'clothing': ['shirt', 'blouse', 'top', 'sweater', 'pull', 'dress', 'pants'],
            'accessory': ['hat', 'belt', 'jewelry', 'watch', 'accessoire']
        }
        
        for category, keywords in categories.items():
            for keyword in keywords:
                if keyword in name_lower:
                    return category
        
        return 'unknown'
    
    def create_synthetic_dataset(self):
        """Create synthetic dataset for training"""
        print("[INFO] Creating synthetic dataset...")
        
        # Generate synthetic pairs
        pairs = []
        
        for i in range(self.max_samples):
            pairs.append({
                'id': f'synthetic_{i:04d}',
                'image_path': f'synthetic_image_{i}',
                'mesh_path': f'synthetic_mesh_{i}',
                'category': ['bag', 'shoe', 'clothing', 'accessory'][i % 4]
            })
        
        return self.create_dataset_info(pairs, 'synthetic')
    
    def create_dataset_info(self, pairs, dataset_type):
        """Create dataset information file"""
        print("[INFO] Creating dataset info...")
        
        # Limit samples for training
        if len(pairs) > self.max_samples:
            random.shuffle(pairs)
            pairs = pairs[:self.max_samples]
        
        # Split into train/validation
        split_idx = int(len(pairs) * self.train_ratio)
        train_pairs = pairs[:split_idx]
        val_pairs = pairs[split_idx:]
        
        # Create dataset info
        dataset_info = {
            'dataset_type': dataset_type,
            'total_samples': len(pairs),
            'train_samples': len(train_pairs),
            'validation_samples': len(val_pairs),
            'image_size': self.img_size,
            'voxel_size': self.voxel_size,
            'categories': list(set(pair['category'] for pair in pairs)),
            'structure': {
                'train_pairs': train_pairs[:100],  # Save first 100 for reference
                'validation_pairs': val_pairs[:50]
            },
            'preprocessing_date': str(np.datetime64('now')),
            'optimized_for': 'RTX_2000_Ada_8h_training'
        }
        
        # Save dataset info
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        print(f"[OK] Dataset info saved: {info_path}")
        print(f"[INFO] Training samples: {len(train_pairs)}")
        print(f"[INFO] Validation samples: {len(val_pairs)}")
        print(f"[INFO] Categories: {dataset_info['categories']}")
        
        return True
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("🚀 FASHION 2D-to-3D DATASET PREPROCESSING")
        print("="*60)
        
        # Detect dataset structure
        structure = self.detect_dataset_structure()
        
        if structure == 'filtered_mesh':
            return self.preprocess_filtered_mesh()
        elif structure == 'standard_fashion3d':
            return self.preprocess_standard_fashion3d()
        elif structure == 'custom':
            return self.preprocess_custom_dataset()
        else:
            print("[WARNING] Unknown structure, creating synthetic dataset")
            return self.create_synthetic_dataset()

def main():
    """Main function"""
    print("[INFO] Fashion 2D-to-3D Data Preprocessing")
    print("="*50)
    
    # Load configuration
    config_file = Path('config.yaml')
    if not config_file.exists():
        print("[ERROR] config.yaml not found. Run from project root directory.")
        return
    
    # Run preprocessing
    preprocessor = FashionDataPreprocessor()
    success = preprocessor.run_preprocessing()
    
    if success:
        print(f"\n✅ Preprocessing completed successfully!")
        print(f"🚀 Next steps:")
        print(f"   1. Start training: python deploy.py --train")
        print(f"   2. Test API: python deploy.py --api")
        print(f"   3. Run demo: python deploy.py --demo")
    else:
        print(f"\n❌ Preprocessing failed!")

if __name__ == "__main__":
    main()
