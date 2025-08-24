#!/usr/bin/env python3
"""
Custom preprocessing for filtered_mesh dataset structure
Each directory contains: {ID}_tex.png + model_cleaned.obj + model_cleaned.obj.mtl
"""

import os
import numpy as np
import cv2
from pathlib import Path
import json
import yaml
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class FilteredMeshPreprocessor:
    """Custom preprocessor for your filtered_mesh dataset"""
    
    def __init__(self, config_path='config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_path = Path(self.config['dataset_path'])
        self.output_dir = Path('./data/')
        self.output_dir.mkdir(exist_ok=True)
        
        # Target sizes
        self.img_size = tuple(self.config['model_config']['input_resolution'])
        self.max_samples = 5000  # Limit for 8h training on RTX 2000 Ada
        
    def discover_pairs(self):
        """Discover all valid image-mesh pairs"""
        print("[INFO] Discovering image-mesh pairs...")
        
        pairs = []
        directories = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        
        for dir_path in tqdm(directories, desc="Scanning directories"):
            dir_name = dir_path.name
            
            # Expected files based on the actual structure
            tex_file = dir_path / f"{dir_name}_tex.png"
            obj_file = dir_path / "model_cleaned.obj"
            mtl_file = dir_path / "model_cleaned.obj.mtl"
            
            # Check if both required files exist
            if tex_file.exists() and obj_file.exists():
                pairs.append({
                    'id': dir_name,
                    'image_path': str(tex_file),
                    'mesh_path': str(obj_file),
                    'material_path': str(mtl_file) if mtl_file.exists() else None,
                    'category': self.extract_category_from_path(str(dir_path))
                })
        
        print(f"[OK] Found {len(pairs)} valid pairs")
        return pairs
    
    def extract_category_from_path(self, path):
        """Extract category from item ID (placeholder - could be enhanced)"""
        # For now, categorize by ID ranges (can be improved with actual labels)
        dir_name = Path(path).name
        item_id = int(dir_name.split('-')[0])
        
        # Simple categorization based on ID ranges (this could be improved)
        if 1 <= item_id <= 100:
            return 'accessories'
        elif 101 <= item_id <= 300:
            return 'clothing'
        elif 301 <= item_id <= 500:
            return 'shoes'
        else:
            return 'bags'
    
    def preprocess_image(self, img_path):
        """Preprocess a texture image"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            
            # Normalize to [-1, 1] for GAN
            img = (img.astype(np.float32) / 127.5) - 1.0
            return img
        except Exception as e:
            print(f"[ERROR] Image processing failed for {img_path}: {e}")
            return None
    
    def analyze_mesh(self, mesh_path):
        """Analyze mesh properties (basic analysis)"""
        try:
            # Simple analysis - count vertices and faces
            vertices = 0
            faces = 0
            
            with open(mesh_path, 'r') as f:
                for line in f:
                    if line.startswith('v '):
                        vertices += 1
                    elif line.startswith('f '):
                        faces += 1
            
            return {
                'vertices': vertices,
                'faces': faces,
                'valid': vertices > 0 and faces > 0
            }
        except Exception as e:
            print(f"[ERROR] Mesh analysis failed for {mesh_path}: {e}")
            return {'vertices': 0, 'faces': 0, 'valid': False}
    
    def create_training_split(self, pairs, train_ratio=0.8):
        """Create training and validation splits"""
        print("[INFO] Creating train/validation split...")
        
        # Shuffle pairs
        import random
        random.seed(42)  # Reproducible split
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # Limit samples for 8h training
        if len(shuffled_pairs) > self.max_samples:
            print(f"[INFO] Limiting to {self.max_samples} samples for 8h training")
            shuffled_pairs = shuffled_pairs[:self.max_samples]
        
        # Split
        split_point = int(len(shuffled_pairs) * train_ratio)
        train_pairs = shuffled_pairs[:split_point]
        val_pairs = shuffled_pairs[split_point:]
        
        return train_pairs, val_pairs
    
    def save_dataset_info(self, train_pairs, val_pairs):
        """Save dataset information for training"""
        dataset_info = {
            'dataset_type': 'filtered_mesh',
            'total_samples': len(train_pairs) + len(val_pairs),
            'train_samples': len(train_pairs),
            'validation_samples': len(val_pairs),
            'image_size': self.img_size,
            'categories': list(set(pair['category'] for pair in train_pairs + val_pairs)),
            'structure': {
                'image_format': '{ID}_tex.png',
                'mesh_format': 'model_cleaned.obj',
                'directory_pattern': '{ID}-{variant}'
            },
            'preprocessing_date': str(np.datetime64('now')),
            'optimized_for': 'RTX_2000_Ada_8h_training'
        }
        
        # Save training pairs info
        dataset_info['train_samples_info'] = [
            {
                'id': pair['id'],
                'category': pair['category'],
                'has_material': pair['material_path'] is not None
            } for pair in train_pairs[:100]  # Save first 100 for reference
        ]
        
        # Save to JSON
        info_path = self.output_dir / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2, default=str)
        
        print(f"[OK] Dataset info saved: {info_path}")
        return dataset_info
    
    def create_sample_outputs(self, pairs, num_samples=5):
        """Create sample outputs to verify preprocessing"""
        print("[INFO] Creating sample outputs...")
        
        samples_dir = self.output_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)
        
        sample_info = []
        
        for i, pair in enumerate(pairs[:num_samples]):
            sample_id = pair['id']
            
            # Process image
            img = self.preprocess_image(pair['image_path'])
            if img is not None:
                # Save processed image
                img_save = ((img + 1.0) * 127.5).astype(np.uint8)
                img_path = samples_dir / f'sample_{sample_id}_processed.png'
                cv2.imwrite(str(img_path), cv2.cvtColor(img_save, cv2.COLOR_RGB2BGR))
                
                # Analyze mesh
                mesh_info = self.analyze_mesh(pair['mesh_path'])
                
                sample_info.append({
                    'id': sample_id,
                    'category': pair['category'],
                    'image_shape': img.shape,
                    'mesh_vertices': mesh_info['vertices'],
                    'mesh_faces': mesh_info['faces'],
                    'processed_image': str(img_path)
                })
        
        # Save sample info
        sample_info_path = samples_dir / 'sample_info.json'
        with open(sample_info_path, 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        print(f"[OK] {len(sample_info)} samples processed and saved to {samples_dir}")
        return sample_info
    
    def run_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("🚀 CUSTOM PREPROCESSING FOR FILTERED_MESH DATASET")
        print("="*60)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Output directory: {self.output_dir}")
        
        # 1. Discover pairs
        pairs = self.discover_pairs()
        if not pairs:
            print("[ERROR] No valid pairs found!")
            return
        
        # 2. Create splits
        train_pairs, val_pairs = self.create_training_split(pairs)
        print(f"[INFO] Split: {len(train_pairs)} train, {len(val_pairs)} validation")
        
        # 3. Save dataset info
        dataset_info = self.save_dataset_info(train_pairs, val_pairs)
        
        # 4. Create samples
        sample_info = self.create_sample_outputs(train_pairs)
        
        # 5. Generate summary
        print("\n" + "="*60)
        print("PREPROCESSING SUMMARY")
        print("="*60)
        print(f"✅ Total valid pairs: {len(pairs)}")
        print(f"✅ Training samples: {len(train_pairs)}")
        print(f"✅ Validation samples: {len(val_pairs)}")
        print(f"✅ Categories detected: {dataset_info['categories']}")
        print(f"✅ Ready for 8h fine-tuning on RTX 2000 Ada")
        
        print(f"\n📁 Files created:")
        print(f"   - {self.output_dir}/dataset_info.json")
        print(f"   - {self.output_dir}/samples/ (with {len(sample_info)} examples)")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review samples in {self.output_dir}/samples/")
        print(f"   2. Start API: python deploy.py --api")
        print(f"   3. Test generation: python deploy.py --demo")
        
        return True

def main():
    """Main function"""
    preprocessor = FilteredMeshPreprocessor()
    success = preprocessor.run_preprocessing()
    
    if success:
        print(f"\n✅ Preprocessing completed successfully!")
    else:
        print(f"\n❌ Preprocessing failed!")

if __name__ == "__main__":
    main()
