#!/usr/bin/env python3
"""
Script de déploiement Fashion 2D-to-3D GAN
Optimisé pour RTX 2000 Ada et fine-tuning 8h
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse
import yaml
import time

def check_conda():
    """Vérifier la disponibilité d'Anaconda"""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] Conda detected: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("[ERROR] Anaconda not found. Please install Anaconda first.")
    return False

def setup_environment():
    """Configuration de l'environnement conda"""
    print("[INFO] Setting up conda environment...")
    
    if not check_conda():
        return False
    
    # Créer l'environnement
    env_file = Path('environment.yaml')
    if env_file.exists():
        try:
            print("[INFO] Creating conda environment 'fashion3d'...")
            cmd = ['conda', 'env', 'create', '-f', str(env_file), '--force']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("[OK] Environment 'fashion3d' created successfully")
            else:
                print("[WARNING] Environment might already exist")
                print(f"[INFO] Output: {result.stderr}")
        except Exception as e:
            print(f"[ERROR] Failed to create environment: {e}")
            return False
    else:
        print("[ERROR] environment.yaml not found")
        return False
    
    return True

def test_installation():
    """Tester l'installation et les imports"""
    print("[INFO] Testing installation...")
    
    try:
        # Test des imports de base
        sys.path.append(str(Path(__file__).parent))
        
        print("[INFO] Testing TensorFlow...")
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        print(f"[OK] TensorFlow: {tf.__version__}")
        print(f"[INFO] GPUs detected: {len(gpus)}")
        
        print("[INFO] Testing Fashion3DGAN...")
        from models.fashion_3d_gan import Fashion3DGAN, setup_gpu
        
        setup_gpu()
        gan = Fashion3DGAN()
        print("[OK] Fashion3DGAN initialized successfully")
        
        print("[INFO] Testing mesh utilities...")
        from utils.mesh_utils import voxels_to_mesh, validate_voxel_input
        import numpy as np
        
        test_voxels = np.random.rand(64, 64, 64)
        is_valid = validate_voxel_input(test_voxels)
        print(f"[OK] Mesh utilities working, voxel validation: {is_valid}")
        
        print("[OK] All tests passed!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def analyze_dataset(dataset_path=None):
    """Analyser le dataset Deep Fashion3D V2"""
    print("[INFO] Analyzing dataset...")
    
    # Utiliser le chemin du config ou celui fourni
    if dataset_path is None:
        config_file = Path('config.yaml')
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            dataset_path = config.get('dataset_path', './deep_fashion3d_v2/')
        else:
            dataset_path = './deep_fashion3d_v2/'
    
    try:
        from scripts.analyze_dataset import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(dataset_path)
        report = analyzer.analyze_complete_structure()
        
        if report:
            print("[OK] Dataset analysis completed")
            print(f"[INFO] Found {report['total_files']} files")
            print(f"[INFO] Images: {report['file_categories']['images']}")
            print(f"[INFO] Meshes: {report['file_categories']['meshes']}")
            return True
        else:
            print("[ERROR] Dataset analysis failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Analysis error: {e}")
        return False

def start_api():
    """Démarrer le serveur API"""
    print("[INFO] Starting Fashion 3D API server...")
    
    try:
        # Essayer d'abord avec l'environnement conda
        if check_conda():
            print("[INFO] Starting with conda environment...")
            cmd = ['conda', 'run', '-n', 'fashion3d', 'python', 'api/server.py']
        else:
            print("[INFO] Starting with current Python...")
            cmd = [sys.executable, 'api/server.py']
        
        print("[INFO] API will be available at: http://localhost:8000")
        print("[INFO] API documentation at: http://localhost:8000/docs")
        print("[INFO] Press Ctrl+C to stop the server")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
    except Exception as e:
        print(f"[ERROR] Failed to start API: {e}")
        print("[INFO] Try running: python api/server.py")

def run_preprocessing():
    """Lancer le preprocessing des données"""
    print("[INFO] Starting data preprocessing...")
    
    try:
        if check_conda():
            cmd = ['conda', 'run', '-n', 'fashion3d', 'python', 'scripts/preprocess_data.py']
        else:
            cmd = [sys.executable, 'scripts/preprocess_data.py']
        
        result = subprocess.run(cmd, capture_output=False)
        
        if result.returncode == 0:
            print("[OK] Preprocessing completed")
            return True
        else:
            print("[ERROR] Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Preprocessing error: {e}")
        return False

def start_training(hours=8):
    """Démarrer l'entraînement"""
    print(f"[INFO] Starting training for {hours} hours...")
    
    try:
        if check_conda():
            cmd = ['conda', 'run', '-n', 'fashion3d', 'python', 'scripts/train.py', '--hours', str(hours)]
        else:
            cmd = [sys.executable, 'scripts/train.py', '--hours', str(hours)]
        
        print(f"[INFO] Training optimized for RTX 2000 Ada")
        print(f"[INFO] Duration: {hours} hours")
        print("[INFO] Monitor progress in logs/")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    except Exception as e:
        print(f"[ERROR] Training error: {e}")

def generate_test_image():
    """Générer une image de test pour la démonstration"""
    print("[INFO] Generating test image...")
    
    try:
        import numpy as np
        from PIL import Image
        
        # Créer une image de test simple
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Ajouter un motif simple pour simuler un article de mode
        center_x, center_y = 128, 128
        for y in range(256):
            for x in range(256):
                distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                if 50 < distance < 80:
                    test_image[y, x] = [255, 100, 50]  # Orange circle
        
        # Sauvegarder
        test_path = Path('test_fashion_image.jpg')
        Image.fromarray(test_image).save(test_path)
        
        print(f"[OK] Test image saved: {test_path}")
        return str(test_path)
        
    except Exception as e:
        print(f"[ERROR] Failed to generate test image: {e}")
        return None

def interactive_demo():
    """Démonstration interactive du système"""
    print("[INFO] Starting interactive demo...")
    print("="*50)
    
    # Vérifier que le système est prêt
    print("[INFO] Checking system status...")
    
    if not test_installation():
        print("[ERROR] System not ready. Run --setup first.")
        return
    
    print("[OK] System ready for demo!")
    print("\nDemo options:")
    print("1. Analyze your dataset")
    print("2. Generate test image and process it")
    print("3. Start API server")
    print("4. Run full pipeline test")
    
    while True:
        try:
            choice = input("\nSelect option (1-4) or 'q' to quit: ").strip()
            
            if choice == 'q':
                break
            elif choice == '1':
                path = input("Dataset path (or Enter for default): ").strip()
                analyze_dataset(path if path else None)
            elif choice == '2':
                test_path = generate_test_image()
                if test_path:
                    print(f"[INFO] Use this image to test the API: {test_path}")
            elif choice == '3':
                print("[INFO] Starting API server (Ctrl+C to stop)...")
                start_api()
            elif choice == '4':
                run_full_pipeline_test()
            else:
                print("[WARNING] Invalid option")
                
        except KeyboardInterrupt:
            print("\n[INFO] Demo interrupted")
            break

def run_full_pipeline_test():
    """Tester le pipeline complet"""
    print("[INFO] Running full pipeline test...")
    
    steps = [
        ("System check", test_installation),
        ("Generate test image", lambda: generate_test_image() is not None),
        ("Test mesh utilities", test_mesh_utils),
    ]
    
    for step_name, step_func in steps:
        print(f"\n[INFO] {step_name}...")
        try:
            success = step_func()
            if success:
                print(f"[OK] {step_name} passed")
            else:
                print(f"[ERROR] {step_name} failed")
                return False
        except Exception as e:
            print(f"[ERROR] {step_name} error: {e}")
            return False
    
    print("\n[OK] Full pipeline test completed successfully!")
    return True

def test_mesh_utils():
    """Tester les utilitaires de mesh"""
    try:
        from utils.mesh_utils import MeshProcessor
        import numpy as np
        
        # Créer des voxels de test
        test_voxels = np.random.rand(32, 32, 32)
        test_voxels = (test_voxels > 0.7).astype(np.float32)
        
        # Traiter
        processor = MeshProcessor('high')
        result = processor.process_voxels_to_mesh(test_voxels)
        
        print(f"[INFO] Generated mesh with {result['face_count']} faces")
        return True
        
    except Exception as e:
        print(f"[ERROR] Mesh utilities test failed: {e}")
        return False

def deploy_complete():
    """Déploiement complet automatique"""
    print("="*60)
    print("FASHION 2D-to-3D GAN - COMPLETE DEPLOYMENT")
    print("="*60)
    
    steps = [
        ("Environment setup", setup_environment),
        ("Installation test", test_installation),
        ("Dataset analysis", lambda: analyze_dataset()),
        ("System demo", interactive_demo)
    ]
    
    for step_name, step_func in steps:
        print(f"\n[STEP] {step_name}...")
        try:
            if step_name == "System demo":
                step_func()  # Demo is interactive
            else:
                success = step_func()
                if success:
                    print(f"[OK] {step_name} completed")
                else:
                    print(f"[ERROR] {step_name} failed")
                    if step_name in ["Environment setup", "Installation test"]:
                        print("[ERROR] Critical step failed, stopping deployment")
                        return False
        except Exception as e:
            print(f"[ERROR] {step_name} error: {e}")
    
    print("\n[OK] Deployment completed!")
    print("\nNext steps:")
    print("1. Adjust dataset path in config.yaml")
    print("2. Run: python deploy.py --analyze")
    print("3. Run: python deploy.py --api")
    
    return True

def main():
    """Fonction principale avec CLI"""
    parser = argparse.ArgumentParser(description='Fashion 2D-to-3D GAN Deployment')
    
    parser.add_argument('--setup', action='store_true', help='Setup conda environment')
    parser.add_argument('--test', action='store_true', help='Test installation')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--preprocess', action='store_true', help='Run data preprocessing')
    parser.add_argument('--train', type=float, default=8, help='Start training (hours)')
    parser.add_argument('--demo', action='store_true', help='Interactive demo')
    parser.add_argument('--deploy', action='store_true', help='Complete deployment')
    
    # Dataset options
    parser.add_argument('--dataset-path', help='Path to Deep Fashion3D V2 dataset')
    
    args = parser.parse_args()
    
    if args.deploy:
        deploy_complete()
    elif args.setup:
        setup_environment()
    elif args.test:
        test_installation()
    elif args.analyze:
        analyze_dataset(args.dataset_path)
    elif args.api:
        start_api()
    elif args.preprocess:
        run_preprocessing()
    elif args.train:
        start_training(args.train)
    elif args.demo:
        interactive_demo()
    else:
        print("FASHION 2D-to-3D GAN DEPLOYMENT")
        print("="*50)
        print("Available commands:")
        print("  --deploy       Complete automated deployment")
        print("  --setup        Setup conda environment")
        print("  --test         Test installation")
        print("  --analyze      Analyze dataset structure")
        print("  --api          Start API server")
        print("  --preprocess   Run data preprocessing")
        print("  --train HOURS  Start training")
        print("  --demo         Interactive demo")
        print()
        print("Examples:")
        print("  python deploy.py --deploy")
        print("  python deploy.py --setup")
        print("  python deploy.py --analyze --dataset-path ./my_dataset/")
        print("  python deploy.py --api")
        print("  python deploy.py --demo")

if __name__ == "__main__":
    main()
