#!/usr/bin/env python3
"""
Auto-installation script for Fashion 2D-to-3D GAN
Complete setup and configuration for RTX 2000 Ada
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import yaml
import json

class Fashion3DAutoInstaller:
    """Auto-installer for Fashion 2D-to-3D GAN"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / 'config.yaml'
        self.env_file = self.project_root / 'environment.yaml'
        
        # Detect system
        self.system = platform.system()
        self.is_windows = self.system == 'Windows'
        
        print("🚀 FASHION 2D-to-3D GAN AUTO-INSTALLER")
        print("="*60)
        print(f"System: {self.system}")
        print(f"Project: {self.project_root}")
    
    def check_prerequisites(self):
        """Check system prerequisites"""
        print("\n[STEP 1] Checking prerequisites...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print(f"[ERROR] Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        
        print(f"[OK] Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check for conda
        try:
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[OK] Conda detected: {result.stdout.strip()}")
                self.has_conda = True
            else:
                print("[WARNING] Conda not found, will use pip")
                self.has_conda = False
        except FileNotFoundError:
            print("[WARNING] Conda not found, will use pip")
            self.has_conda = False
        
        # Check for GPU
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                print(f"[OK] {len(gpus)} GPU(s) detected")
                self.has_gpu = True
            else:
                print("[WARNING] No GPU detected - will use CPU")
                self.has_gpu = False
        except ImportError:
            print("[INFO] TensorFlow not installed yet")
            self.has_gpu = False
        
        return True
    
    def setup_conda_environment(self):
        """Setup conda environment"""
        print("\n[STEP 2] Setting up conda environment...")
        
        if not self.has_conda:
            print("[INFO] Skipping conda setup, using pip")
            return True
        
        try:
            # Create environment
            print("[INFO] Creating conda environment 'fashion3d'...")
            cmd = ['conda', 'env', 'create', '-f', str(self.env_file), '--force']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("[OK] Conda environment created successfully")
                return True
            else:
                print(f"[WARNING] Conda environment creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Conda setup failed: {e}")
            return False
    
    def install_pip_dependencies(self):
        """Install pip dependencies"""
        print("\n[STEP 3] Installing pip dependencies...")
        
        # Core dependencies
        core_deps = [
            'tensorflow>=2.13.0',
            'numpy>=1.24.0',
            'opencv-python>=4.8.0',
            'pillow>=10.0.0',
            'scikit-image>=0.21.0',
            'fastapi>=0.103.0',
            'uvicorn>=0.23.0',
            'python-multipart>=0.0.6',
            'pyyaml>=6.0',
            'tqdm>=4.65.0',
            'requests>=2.31.0'
        ]
        
        # 3D processing dependencies
        mesh_deps = [
            'trimesh>=3.23.0',
            'open3d>=0.17.0',
            'mcubes>=0.1.0',
            'plotly>=5.16.0',
            'pyvista>=0.42.0'
        ]
        
        # Development dependencies
        dev_deps = [
            'jupyter>=1.0.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'rich>=13.5.0',
            'typer>=0.9.0'
        ]
        
        all_deps = core_deps + mesh_deps + dev_deps
        
        try:
            for dep in all_deps:
                print(f"[INFO] Installing {dep}...")
                cmd = [sys.executable, '-m', 'pip', 'install', dep]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"[WARNING] Failed to install {dep}: {result.stderr}")
                else:
                    print(f"[OK] {dep} installed")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Pip installation failed: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("\n[STEP 4] Creating project directories...")
        
        directories = [
            'data',
            'outputs',
            'logs',
            'models/checkpoints',
            'temp',
            'samples'
        ]
        
        for dir_name in directories:
            dir_path = self.project_root / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Created {dir_path}")
        
        return True
    
    def setup_configuration(self):
        """Setup configuration files"""
        print("\n[STEP 5] Setting up configuration...")
        
        # Update config with detected settings
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {}
        
        # Update GPU settings
        config['gpu_config'] = {
            'memory_growth': True,
            'mixed_precision': True,
            'batch_size': 6 if self.has_gpu else 2,
            'num_workers': 4 if self.has_gpu else 2
        }
        
        # Update model settings
        config['model_config'] = {
            'input_resolution': [256, 256],
            'voxel_resolution': 64,
            'latent_dim': 512,
            'generator_lr': 0.0002,
            'discriminator_lr': 0.0001,
            'beta1': 0.5,
            'beta2': 0.999
        }
        
        # Update training settings
        config['training_config'] = {
            'max_hours': 8,
            'epochs_per_hour': 15,
            'save_interval': 30,
            'validation_interval': 60,
            'early_stopping_patience': 20
        }
        
        # Save updated config
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"[OK] Configuration updated: {self.config_file}")
        return True
    
    def create_startup_scripts(self):
        """Create startup scripts for easy launching"""
        print("\n[STEP 6] Creating startup scripts...")
        
        if self.is_windows:
            # Create Windows batch files
            self._create_windows_scripts()
        else:
            # Create Unix shell scripts
            self._create_unix_scripts()
        
        return True
    
    def _create_windows_scripts(self):
        """Create Windows batch scripts"""
        
        # API startup script
        api_script = '''@echo off
echo Starting Fashion 3D API Server...
cd /d "%~dp0"
if exist "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe" (
    conda activate fashion3d
    python api/server.py
) else (
    python api/server.py
)
pause
'''
        
        with open(self.project_root / 'start_api.bat', 'w') as f:
            f.write(api_script)
        
        # Training startup script
        train_script = '''@echo off
echo Starting Fashion 3D Training...
cd /d "%~dp0"
if exist "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe" (
    conda activate fashion3d
    python scripts/train.py --hours 8
) else (
    python scripts/train.py --hours 8
)
pause
'''
        
        with open(self.project_root / 'start_training.bat', 'w') as f:
            f.write(train_script)
        
        # Demo startup script
        demo_script = '''@echo off
echo Starting Fashion 3D Demo...
cd /d "%~dp0"
if exist "C:\\ProgramData\\Anaconda3\\Scripts\\conda.exe" (
    conda activate fashion3d
    python deploy.py --demo
) else (
    python deploy.py --demo
)
pause
'''
        
        with open(self.project_root / 'start_demo.bat', 'w') as f:
            f.write(demo_script)
        
        print("[OK] Windows startup scripts created")
    
    def _create_unix_scripts(self):
        """Create Unix shell scripts"""
        
        # API startup script
        api_script = '''#!/bin/bash
echo "Starting Fashion 3D API Server..."
cd "$(dirname "$0")"
if command -v conda &> /dev/null; then
    conda activate fashion3d
    python api/server.py
else
    python api/server.py
fi
'''
        
        api_path = self.project_root / 'start_api.sh'
        with open(api_path, 'w') as f:
            f.write(api_script)
        os.chmod(api_path, 0o755)
        
        # Training startup script
        train_script = '''#!/bin/bash
echo "Starting Fashion 3D Training..."
cd "$(dirname "$0")"
if command -v conda &> /dev/null; then
    conda activate fashion3d
    python scripts/train.py --hours 8
else
    python scripts/train.py --hours 8
fi
'''
        
        train_path = self.project_root / 'start_training.sh'
        with open(train_path, 'w') as f:
            f.write(train_script)
        os.chmod(train_path, 0o755)
        
        print("[OK] Unix startup scripts created")
    
    def run_initial_tests(self):
        """Run initial tests to verify installation"""
        print("\n[STEP 7] Running initial tests...")
        
        try:
            # Test imports
            print("[INFO] Testing imports...")
            import tensorflow as tf
            import numpy as np
            import cv2
            from PIL import Image
            import fastapi
            import yaml
            
            print("[OK] Core imports successful")
            
            # Test GPU
            if self.has_gpu:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                print(f"[OK] GPU detection: {len(gpus)} GPU(s)")
            
            # Test model creation
            print("[INFO] Testing model creation...")
            sys.path.append(str(self.project_root))
            from models.fashion_3d_gan import Fashion3DGAN, setup_gpu
            
            setup_gpu()
            gan = Fashion3DGAN()
            print("[OK] Model creation successful")
            
            # Test mesh utilities
            print("[INFO] Testing mesh utilities...")
            from utils.mesh_utils import MeshProcessor
            
            processor = MeshProcessor('high')
            test_voxels = np.random.rand(32, 32, 32)
            result = processor.process_voxels_to_mesh(test_voxels)
            print(f"[OK] Mesh processing: {result['face_count']} faces")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Test failed: {e}")
            return False
    
    def create_dataset_info(self):
        """Create initial dataset info"""
        print("\n[STEP 8] Creating dataset info...")
        
        dataset_info = {
            'dataset_name': 'Synthetic Fashion Dataset',
            'total_samples': 1000,
            'structure': 'synthetic',
            'description': 'Synthetic dataset for initial testing and training',
            'categories': ['bags', 'shoes', 'clothing', 'accessories'],
            'file_structure': {
                'folders': 0,
                'images_per_folder': 0,
                'meshes_per_folder': 0,
                'image_format': 'Synthetic',
                'mesh_format': 'Synthetic'
            },
            'pairing_quality': 'Synthetic pairs',
            'training_ready': True,
            'estimated_training_time': '6-8 hours on RTX 2000 Ada'
        }
        
        info_path = self.project_root / 'data' / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"[OK] Dataset info created: {info_path}")
        return True
    
    def run_complete_installation(self):
        """Run complete installation process"""
        print("🚀 Starting complete installation...")
        
        steps = [
            ("Checking prerequisites", self.check_prerequisites),
            ("Setting up conda environment", self.setup_conda_environment),
            ("Installing pip dependencies", self.install_pip_dependencies),
            ("Creating directories", self.create_directories),
            ("Setting up configuration", self.setup_configuration),
            ("Creating startup scripts", self.create_startup_scripts),
            ("Creating dataset info", self.create_dataset_info),
            ("Running initial tests", self.run_initial_tests)
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            print(f"[STEP] {step_name}")
            print('='*60)
            
            try:
                success = step_func()
                if success:
                    print(f"[OK] {step_name} completed")
                else:
                    print(f"[ERROR] {step_name} failed")
                    return False
            except Exception as e:
                print(f"[ERROR] {step_name} error: {e}")
                return False
        
        print(f"\n{'='*60}")
        print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print('='*60)
        
        self.print_next_steps()
        return True
    
    def print_next_steps(self):
        """Print next steps for the user"""
        print("\n🚀 NEXT STEPS:")
        print("1. Start the API server:")
        if self.is_windows:
            print("   Double-click: start_api.bat")
        else:
            print("   ./start_api.sh")
        
        print("\n2. Test the system:")
        if self.is_windows:
            print("   Double-click: start_demo.bat")
        else:
            print("   python deploy.py --demo")
        
        print("\n3. Start training:")
        if self.is_windows:
            print("   Double-click: start_training.bat")
        else:
            print("   ./start_training.sh")
        
        print("\n4. Web interface:")
        print("   http://localhost:8000 (API info)")
        print("   http://localhost:8000/docs (Interactive testing)")
        
        print("\n5. Upload your own dataset:")
        print("   - Place your dataset in the specified folder")
        print("   - Update config.yaml with the correct path")
        print("   - Run: python deploy.py --analyze")
        print("   - Run: python scripts/preprocess_data.py")
        
        print(f"\n📁 Project structure:")
        print(f"   - Configuration: {self.config_file}")
        print(f"   - API Server: api/server.py")
        print(f"   - Models: models/fashion_3d_gan.py")
        print(f"   - Training: scripts/train.py")
        print(f"   - Documentation: README.md")

def main():
    """Main function"""
    installer = Fashion3DAutoInstaller()
    
    try:
        success = installer.run_complete_installation()
        
        if success:
            print("\n✅ Installation completed successfully!")
            print("🎨 Your Fashion 2D-to-3D GAN is ready to use!")
        else:
            print("\n❌ Installation failed!")
            print("Please check the error messages above and try again.")
            
    except KeyboardInterrupt:
        print("\n[INFO] Installation interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Installation error: {e}")

if __name__ == "__main__":
    main()
