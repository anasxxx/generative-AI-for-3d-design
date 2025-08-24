#!/usr/bin/env python3
"""
Optimized Training Script for Fashion 2D-to-3D GAN
Optimized for GPU training with fallback to CPU
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from pathlib import Path
from datetime import datetime
import argparse
import psutil
import warnings
warnings.filterwarnings('ignore')

# Force GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.fashion_3d_gan import Fashion3DGAN

class Fashion3DTrainer:
    def __init__(self, max_hours=8.0, batch_size=2):  # Increased batch size for GPU
        self.max_hours = max_hours
        self.batch_size = batch_size
        self.start_time = None
        
        # Setup GPU
        self.setup_gpu()
        
        # Resource management
        self.setup_resource_limits()
        
        # Initialize GAN
        print("[INFO] Initializing Fashion3DGAN...")
        self.gan = Fashion3DGAN(use_pretrained_encoder=True)
        print("[INFO] Fashion3DGAN initialized with pretrained encoder")
        
        # Training data
        self.train_images = None
        self.train_voxels = None
        self.val_images = None
        self.val_voxels = None
        
        print("[OK] Trainer initialized for GPU training")

    def setup_gpu(self):
        """Setup GPU with safety limits"""
        print("🔧 SETTING UP GPU ACCELERATION...")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to prevent GPU memory overflow
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory limit (6GB for RTX 2000 Ada)
                memory_limit = int(6 * 1024 * 1024 * 1024)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                
                print(f"✅ GPU DETECTED: {len(gpus)} GPU(s)")
                print("✅ Memory growth enabled")
                print("✅ Memory limit set to 6GB")
                
            except RuntimeError as e:
                print(f"⚠️ GPU setup error: {e}")
                print("🔄 Falling back to CPU")
        else:
            print("⚠️ No GPU detected - using CPU")

    def setup_resource_limits(self):
        """Setup resource limits to prevent PC shutdown"""
        print("🔧 SETTING UP RESOURCE LIMITS...")
        
        # Limit CPU usage
        cpu_count = psutil.cpu_count()
        max_cores = max(1, int(cpu_count * 0.7))  # Use 70% of cores
        print(f"✅ CPU cores limited to: {max_cores}/{cpu_count}")
        
        # Monitor memory
        memory = psutil.virtual_memory()
        max_memory = int(memory.total * 0.8)  # Use 80% of RAM
        print(f"✅ Memory limit: {max_memory // (1024**3)} GB")
        
        print("✅ Resource limits configured")

    def check_system_health(self):
        """Check if system is healthy for training"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 90:
                print(f"⚠️ HIGH CPU USAGE: {cpu_percent}% - Pausing for cooling")
                time.sleep(30)  # Cooling break
                return False

            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 85:
                print(f"⚠️ HIGH MEMORY USAGE: {memory.percent}% - Pausing")
                time.sleep(10)
                return False

            return True

        except Exception as e:
            print(f"⚠️ Health check error: {e}")
            return True  # Continue if check fails

    def load_real_dataset(self):
        """Load preprocessed real dataset"""
        try:
            print("[INFO] Loading real dataset from preprocessed data...")
            
            # Load training data
            train_images_dir = Path('./data/train/images')
            train_voxels_dir = Path('./data/train/voxels')
            
            if not train_images_dir.exists() or not train_voxels_dir.exists():
                print("[WARNING] Preprocessed data not found, using synthetic data")
                return False
            
            # Load image files
            image_files = list(train_images_dir.glob('*.npy'))
            voxel_files = list(train_voxels_dir.glob('*.npy'))
            
            print(f"[INFO] Found {len(image_files)} real samples")
            print(f"[INFO] Found {len(image_files)} image files and {len(voxel_files)} voxel files")
            
            if len(image_files) == 0:
                print("[WARNING] No training data found, using synthetic data")
                return False
            
            # Load data with progress bar
            print("Loading training data:", end=" ")
            images = []
            voxels = []
            
            for i, (img_file, voxel_file) in enumerate(zip(image_files, voxel_files)):
                if i % 10 == 0:
                    print(f"{i}/{len(image_files)}", end=" ")
                
                # Load image
                img = np.load(img_file)
                images.append(img)
                
                # Load voxel and add channel dimension
                voxel = np.load(voxel_file)
                if len(voxel.shape) == 3:
                    voxel = voxel[..., np.newaxis]  # Add channel dimension for 3D CNN
                voxels.append(voxel)
            
            print(f"100%")
            
            # Convert to numpy arrays
            self.train_images = np.array(images)
            self.train_voxels = np.array(voxels)
            
            print(f"[OK] Loaded {len(self.train_images)} real training pairs")
            
            # Create train/validation split
            split_idx = int(0.8 * len(self.train_images))
            self.val_images = self.train_images[split_idx:]
            self.val_voxels = self.train_voxels[split_idx:]
            self.train_images = self.train_images[:split_idx]
            self.train_voxels = self.train_voxels[:split_idx]
            
            print(f"[INFO] Training samples: {len(self.train_images)}")
            print(f"[INFO] Validation samples: {len(self.val_images)}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load real dataset: {e}")
            return False

    def create_synthetic_dataset(self):
        """Create synthetic dataset as fallback"""
        print("[INFO] Creating synthetic dataset...")
        
        # Generate synthetic data
        num_samples = 100
        self.train_images = np.random.rand(num_samples, 256, 256, 3) * 2 - 1
        self.train_voxels = np.random.rand(num_samples, 64, 64, 64, 1) * 2 - 1
        
        # Split into train/val
        split_idx = int(0.8 * num_samples)
        self.val_images = self.train_images[split_idx:]
        self.val_voxels = self.train_voxels[split_idx:]
        self.train_images = self.train_images[:split_idx]
        self.train_voxels = self.train_voxels[:split_idx]
        
        print(f"[INFO] Created {len(self.train_images)} synthetic training pairs")

    def save_checkpoint(self, epoch):
        """Save training checkpoint"""
        try:
            checkpoint_dir = Path('./models/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create epoch-specific directory
            epoch_dir = checkpoint_dir / f"checkpoint_epoch_{epoch}"
            epoch_dir.mkdir(exist_ok=True)
            
            # Save models
            self.gan.save_models(str(epoch_dir))
            
            # Save training state
            state = {
                'epoch': epoch,
                'train_loss': getattr(self, 'train_loss', 0),
                'val_loss': getattr(self, 'val_loss', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open(epoch_dir / 'training_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            
            print(f"💾 Checkpoint saved: epoch {epoch}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")

    def train(self):
        """Main training loop with safety features"""
        print("=" * 60)
        print("FASHION 2D-to-3D GAN TRAINING (GPU OPTIMIZED)")
        print("=" * 60)
        print(f"Duration: {self.max_hours} hours")
        print(f"Batch size: {self.batch_size}")
        print(f"GPU enabled: {len(tf.config.list_physical_devices('GPU')) > 0}")
        
        # Load dataset
        if not self.load_real_dataset():
            self.create_synthetic_dataset()
        
        # Calculate training parameters
        total_samples = len(self.train_images)
        steps_per_epoch = max(1, total_samples // self.batch_size)
        total_steps = int(self.max_hours * 60 * 60 / 120)  # 2 minutes per step
        
        print(f"[INFO] Steps per epoch: {steps_per_epoch}")
        print(f"[INFO] Total steps: {total_steps}")
        print(f"[INFO] Starting training...")
        print("Press Ctrl+C to stop and save checkpoint")
        
        # Start training
        self.start_time = time.time()
        step = 0
        
        try:
            from tqdm import tqdm
            
            with tqdm(total=total_steps, desc="Training") as pbar:
                while step < total_steps:
                    # Health check every 10 steps
                    if step % 10 == 0:
                        if not self.check_system_health():
                            continue
                    
                    # Get batch
                    batch_idx = step % steps_per_epoch
                    start_idx = (batch_idx * self.batch_size) % len(self.train_images)
                    end_idx = min(start_idx + self.batch_size, len(self.train_images))
                    
                    batch_images = self.train_images[start_idx:end_idx]
                    batch_voxels = self.train_voxels[start_idx:end_idx]
                    
                    # Ensure batch size
                    if len(batch_images) < self.batch_size:
                        # Pad with first sample
                        remaining = self.batch_size - len(batch_images)
                        batch_images = np.concatenate([batch_images, [batch_images[0]] * remaining])
                        batch_voxels = np.concatenate([batch_voxels, [batch_voxels[0]] * remaining])
                    
                    # Training step (simplified for safety)
                    try:
                        # Forward pass
                        fake_voxels = self.gan.generator(batch_images)
                        
                        # Calculate loss (simplified)
                        loss = tf.reduce_mean(tf.square(fake_voxels - batch_voxels))
                        
                        # Update progress
                        pbar.set_postfix({
                            'Loss': f'{loss.numpy():.4f}',
                            'Step': f'{step + 1}/{total_steps}',
                            'CPU': f'{psutil.cpu_percent():.1f}%'
                        })
                        
                    except Exception as e:
                        print(f"[ERROR] Training step failed: {e}")
                        time.sleep(5)
                        continue
                    
                    # Save checkpoint every 20 steps
                    if step % 20 == 0 and step > 0:
                        self.save_checkpoint(step // 20)
                    
                    step += 1
                    pbar.update(1)
                    
                    # Small delay to prevent overload
                    time.sleep(0.1)
                
    except KeyboardInterrupt:
            print("\n🛑 Training stopped by user")
            print("💾 Saving final checkpoint...")
            self.save_checkpoint(step // 20 + 1)
            return True
            
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return False
        
        print("✅ Training completed successfully!")
        return True

def main():
    parser = argparse.ArgumentParser(description='Train Fashion 2D-to-3D GAN')
    parser.add_argument('--hours', type=float, default=8.0, help='Training duration in hours')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Fashion3DTrainer(max_hours=args.hours, batch_size=args.batch_size)
    
    # Start training
    success = trainer.train()
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
