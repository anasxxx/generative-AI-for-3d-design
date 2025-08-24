#!/usr/bin/env python3
"""
Script d'entraînement Fashion 2D-to-3D GAN
Optimisé pour fine-tuning 8h sur RTX 2000 Ada
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
import yaml
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.fashion_3d_gan import Fashion3DGAN, setup_gpu
from utils.mesh_utils import MeshProcessor

class Fashion3DTrainer:
    """Trainer class for Fashion 2D-to-3D GAN"""
    
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup GPU
        setup_gpu()
        
        # Initialize GAN
        self.gan = Fashion3DGAN(
            img_shape=tuple(self.config['model_config']['input_resolution']) + (3,),
            voxel_size=self.config['model_config']['voxel_resolution'],
            latent_dim=self.config['model_config']['latent_dim'],
            learning_rates=(
                self.config['model_config']['generator_lr'],
                self.config['model_config']['discriminator_lr']
            )
        )
        
        # Training parameters
        self.batch_size = self.config['gpu_config']['batch_size']
        self.max_hours = self.config['training_config']['max_hours']
        self.save_interval = self.config['training_config']['save_interval']
        self.validation_interval = self.config['training_config']['validation_interval']
        
        # Directories
        self.checkpoint_dir = Path('./models/checkpoints/')
        self.logs_dir = Path('./logs/')
        self.output_dir = Path('./outputs/')
        
        for dir_path in [self.checkpoint_dir, self.logs_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load dataset info
        dataset_info_path = Path('./data/dataset_info.json')
        if dataset_info_path.exists():
            with open(dataset_info_path, 'r') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = {'total_samples': 1000}  # Fallback
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history = {
            'gen_loss': [],
            'disc_loss': [],
            'reconstruction_loss': [],
            'validation_metrics': []
        }
        
        print(f"[OK] Trainer initialized for {self.dataset_info['total_samples']} samples")
    
    def load_real_dataset(self):
        """Load real dataset from preprocessed data"""
        print(f"[INFO] Loading real dataset from preprocessed data...")
        
        # Load dataset info
        dataset_info_path = Path('./data/dataset_info.json')
        if not dataset_info_path.exists():
            raise FileNotFoundError("Dataset info not found. Run preprocessing first.")
        
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        print(f"[INFO] Found {dataset_info['total_samples']} real samples")
        
        # Load training data
        train_data_path = Path('./data/train/')
        if not train_data_path.exists():
            raise FileNotFoundError("Training data not found. Run preprocessing first.")
        
        # Load image and voxel pairs
        images = []
        voxels = []
        
        # Load from numpy files
        image_files = list(train_data_path.glob('images/*.npy'))
        voxel_files = list(train_data_path.glob('voxels/*.npy'))
        
        print(f"[INFO] Found {len(image_files)} image files and {len(voxel_files)} voxel files")
        
        # Load pairs
        for img_file in tqdm(image_files, desc="Loading training data"):
            # Find corresponding voxel file
            voxel_file = train_data_path / 'voxels' / img_file.name
            
            if voxel_file.exists():
                try:
                    # Load image and voxel
                    img = np.load(img_file)
                    voxel = np.load(voxel_file)
                    
                    # Ensure correct shapes
                    if img.shape == (256, 256, 3) and voxel.shape == (64, 64, 64):
                        images.append(img)
                        # Add channel dimension to voxel for 3D CNN
                        voxel_with_channel = voxel[..., np.newaxis]  # Shape: (64, 64, 64, 1)
                        voxels.append(voxel_with_channel)
                    elif img.shape == (256, 256, 3) and voxel.shape == (64, 64, 64, 1):
                        # Already has channel dimension
                        images.append(img)
                        voxels.append(voxel)
                except Exception as e:
                    print(f"[WARNING] Failed to load {img_file.name}: {e}")
                    continue
        
        if len(images) == 0:
            raise ValueError("No valid image-voxel pairs found!")
        
        print(f"[OK] Loaded {len(images)} real training pairs")
        return np.array(images), np.array(voxels)
    
    def create_synthetic_dataset(self, num_samples=1000):
        """Create synthetic dataset for training (fallback only)"""
        print(f"[WARNING] Creating synthetic dataset with {num_samples} samples...")
        print("[WARNING] This is a fallback - real data should be used for production!")
        
        # Generate synthetic image-mesh pairs
        images = []
        voxels = []
        
        for i in tqdm(range(num_samples), desc="Generating synthetic data"):
            # Generate random fashion-like image
            img = np.random.rand(256, 256, 3) * 2 - 1  # [-1, 1] range
            
            # Add some fashion-like patterns
            center_x, center_y = 128, 128
            y, x = np.ogrid[:256, :256]
            
            # Create different fashion item shapes
            item_type = i % 4  # 4 types: bag, shoe, clothing, accessory
            
            if item_type == 0:  # Bag
                # Create bag-like shape
                mask = ((x - center_x)**2 / 60**2 + (y - center_y)**2 / 40**2) < 1
                img[mask] = np.random.rand(3) * 0.5 + 0.25
            elif item_type == 1:  # Shoe
                # Create shoe-like shape
                mask = ((x - center_x)**2 / 50**2 + (y - center_y)**2 / 30**2) < 1
                img[mask] = np.random.rand(3) * 0.5 + 0.25
            elif item_type == 2:  # Clothing
                # Create clothing-like shape
                mask = ((x - center_x)**2 / 80**2 + (y - center_y)**2 / 60**2) < 1
                img[mask] = np.random.rand(3) * 0.5 + 0.25
            else:  # Accessory
                # Create accessory-like shape
                mask = ((x - center_x)**2 / 30**2 + (y - center_y)**2 / 30**2) < 1
                img[mask] = np.random.rand(3) * 0.5 + 0.25
            
            # Generate corresponding 3D voxel
            voxel = np.random.rand(64, 64, 64) * 0.1  # Low density background
            
            # Add some 3D structure
            center_z = 32
            for z in range(64):
                if abs(z - center_z) < 15:  # Create volume
                    # Add fashion item volume
                    y_coords, x_coords = np.ogrid[:64, :64]
                    mask_3d = ((x_coords - 32)**2 / 20**2 + (y_coords - 32)**2 / 15**2) < 1
                    voxel[z][mask_3d] = np.random.rand() * 0.8 + 0.2
            
            images.append(img)
            voxels.append(voxel)
        
        return np.array(images), np.array(voxels)
    
    @tf.function
    def train_step(self, real_images, real_voxels):
        """Single training step"""
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake voxels
            fake_voxels, latent = self.gan.generator(real_images, training=True)
            
            # Discriminator predictions
            real_output = self.gan.discriminator(real_voxels, training=True)
            fake_output = self.gan.discriminator(fake_voxels, training=True)
            
            # Calculate losses
            gen_loss = self.gan.adversarial_loss(tf.ones_like(fake_output), fake_output)
            disc_loss_real = self.gan.adversarial_loss(tf.ones_like(real_output), real_output)
            disc_loss_fake = self.gan.adversarial_loss(tf.zeros_like(fake_output), fake_output)
            disc_loss = disc_loss_real + disc_loss_fake
            
            # Reconstruction loss
            reconstruction_loss = self.gan.reconstruction_loss(real_voxels, fake_voxels)
            
            # Total generator loss
            total_gen_loss = gen_loss + 10.0 * reconstruction_loss
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(total_gen_loss, self.gan.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.gan.discriminator.trainable_variables)
        
        # Apply gradients
        self.gan.gen_optimizer.apply_gradients(zip(gen_gradients, self.gan.generator.trainable_variables))
        self.gan.disc_optimizer.apply_gradients(zip(disc_gradients, self.gan.discriminator.trainable_variables))
        
        return total_gen_loss, disc_loss, reconstruction_loss
    
    def validate(self, val_images, val_voxels):
        """Validation step"""
        fake_voxels, _ = self.gan.generator(val_images, training=False)
        
        # Calculate validation metrics
        reconstruction_loss = self.gan.reconstruction_loss(val_voxels, fake_voxels)
        
        # Calculate voxel occupancy
        occupancy = tf.reduce_mean(tf.cast(fake_voxels > 0.5, tf.float32))
        
        return {
            'reconstruction_loss': float(reconstruction_loss),
            'voxel_occupancy': float(occupancy)
        }
    
    def save_checkpoint(self, epoch, metrics):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}'
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model weights
        self.gan.save_models(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'best_loss': self.best_loss,
            'training_history': self.training_history,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        state_path = checkpoint_path / 'training_state.json'
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2, default=str)
        
        print(f"[OK] Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        try:
            # Load model weights
            self.gan.load_models(str(checkpoint_path))
            
            # Load training state
            state_path = checkpoint_path / 'training_state.json'
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.current_epoch = state['epoch']
                self.best_loss = state['best_loss']
                self.training_history = state['training_history']
                
                print(f"[OK] Checkpoint loaded from epoch {self.current_epoch}")
                return True
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
        
        return False
    
    def train(self, resume_from=None):
        """Main training loop"""
        print("="*60)
        print("FASHION 2D-to-3D GAN TRAINING")
        print("="*60)
        print(f"Duration: {self.max_hours} hours")
        print(f"Batch size: {self.batch_size}")
        print(f"Samples: {self.dataset_info['total_samples']}")
        print("Optimized for RTX 2000 Ada")
        
        # Load checkpoint if specified
        if resume_from:
            checkpoint_path = Path(resume_from)
            if self.load_checkpoint(checkpoint_path):
                print(f"[INFO] Resuming from epoch {self.current_epoch}")
            else:
                print("[WARNING] Failed to load checkpoint, starting fresh")
        
        # Load real dataset
        try:
            train_images, train_voxels = self.load_real_dataset()
        except Exception as e:
            print(f"[WARNING] Failed to load real dataset: {e}")
            print("[INFO] Falling back to synthetic data for demo...")
            num_samples = min(self.dataset_info['total_samples'], 2000)  # Limit for demo
            train_images, train_voxels = self.create_synthetic_dataset(num_samples)
        
        # Split into train/validation
        split_idx = int(0.8 * len(train_images))
        val_images = train_images[split_idx:]
        val_voxels = train_voxels[split_idx:]
        train_images = train_images[:split_idx]
        train_voxels = train_voxels[:split_idx]
        
        print(f"[INFO] Training samples: {len(train_images)}")
        print(f"[INFO] Validation samples: {len(val_images)}")
        
        # Calculate steps per epoch and total steps
        steps_per_epoch = len(train_images) // self.batch_size
        total_steps = int(self.max_hours * 60 / 2)  # 2 minutes per epoch estimate
        
        print(f"[INFO] Steps per epoch: {steps_per_epoch}")
        print(f"[INFO] Total steps: {total_steps}")
        
        # Training loop
        start_time = time.time()
        last_save_time = time.time()
        last_val_time = time.time()
        
        print(f"\n[INFO] Starting training...")
        print("Press Ctrl+C to stop and save checkpoint")
        
        try:
            for step in tqdm(range(total_steps), desc="Training"):
                # Create batch
                batch_idx = (step * self.batch_size) % len(train_images)
                batch_images = train_images[batch_idx:batch_idx + self.batch_size]
                batch_voxels = train_voxels[batch_idx:batch_idx + self.batch_size]
                
                # Ensure batch size
                if len(batch_images) < self.batch_size:
                    # Wrap around
                    remaining = self.batch_size - len(batch_images)
                    batch_images = np.concatenate([batch_images, train_images[:remaining]])
                    batch_voxels = np.concatenate([batch_voxels, train_voxels[:remaining]])
                
                # Training step
                gen_loss, disc_loss, recon_loss = self.train_step(batch_images, batch_voxels)
                
                # Update metrics
                self.training_history['gen_loss'].append(float(gen_loss))
                self.training_history['disc_loss'].append(float(disc_loss))
                self.training_history['reconstruction_loss'].append(float(recon_loss))
                
                # Validation
                current_time = time.time()
                if current_time - last_val_time > self.validation_interval * 60:
                    val_metrics = self.validate(val_images[:self.batch_size], val_voxels[:self.batch_size])
                    self.training_history['validation_metrics'].append(val_metrics)
                    last_val_time = current_time
                    
                    print(f"\n[VAL] Epoch {step//steps_per_epoch + 1}, "
                          f"Recon Loss: {val_metrics['reconstruction_loss']:.4f}, "
                          f"Occupancy: {val_metrics['voxel_occupancy']:.3f}")
                
                # Save checkpoint
                if current_time - last_save_time > self.save_interval * 60:
                    metrics = {
                        'gen_loss': float(gen_loss),
                        'disc_loss': float(disc_loss),
                        'reconstruction_loss': float(recon_loss)
                    }
                    self.save_checkpoint(step//steps_per_epoch + 1, metrics)
                    last_save_time = current_time
                
                # Check if we should stop
                elapsed_hours = (time.time() - start_time) / 3600
                if elapsed_hours >= self.max_hours:
                    print(f"\n[INFO] Training completed after {elapsed_hours:.1f} hours")
                    break
                
        except KeyboardInterrupt:
            print(f"\n[INFO] Training interrupted by user")
        
        # Final save
        final_metrics = {
            'gen_loss': float(gen_loss),
            'disc_loss': float(disc_loss),
            'reconstruction_loss': float(recon_loss)
        }
        self.save_checkpoint(step//steps_per_epoch + 1, final_metrics)
        
        # Save final model
        final_model_path = self.checkpoint_dir / 'final_model'
        self.gan.save_models(str(final_model_path))
        
        print(f"\n[OK] Training completed!")
        print(f"[INFO] Final model saved: {final_model_path}")
        print(f"[INFO] Training history: {len(self.training_history['gen_loss'])} steps")
        
        return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Fashion 2D-to-3D GAN Training')
    parser.add_argument('--hours', type=float, default=8, help='Training duration in hours')
    parser.add_argument('--resume', help='Resume from checkpoint path')
    
    args = parser.parse_args()
    
    # Update config with command line args
    config_file = Path('config.yaml')
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['training_config']['max_hours'] = args.hours
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
    
    # Start training
    trainer = Fashion3DTrainer()
    success = trainer.train(resume_from=args.resume)
    
    if success:
        print(f"\n✅ Training completed successfully!")
        print(f"🚀 Next steps:")
        print(f"   1. Test the model: python deploy.py --demo")
        print(f"   2. Start API: python deploy.py --api")
        print(f"   3. Check outputs in ./outputs/")
    else:
        print(f"\n❌ Training failed!")

if __name__ == "__main__":
    main()
