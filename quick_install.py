#!/usr/bin/env python3
"""
Quick installation script for Fashion 2D-to-3D GAN
Fixes dependency issues and installs missing packages
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n[INFO] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {description} completed")
            return True
        else:
            print(f"[WARNING] {description} had issues:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"[ERROR] {description} failed: {e}")
        return False

def install_core_dependencies():
    """Install core dependencies that are missing"""
    print("="*60)
    print("INSTALLING CORE DEPENDENCIES")
    print("="*60)
    
    # Core packages
    core_packages = [
        "conda install -c conda-forge tensorflow=2.13.0 -y",
        "conda install -c conda-forge tqdm numpy scipy matplotlib pillow -y", 
        "conda install -c conda-forge opencv scikit-image h5py pyyaml -y",
        "conda install -c conda-forge fastapi uvicorn python-multipart -y",
        "conda install -c conda-forge jupyter ipykernel -y"
    ]
    
    for cmd in core_packages:
        run_command(cmd, f"Installing: {cmd.split()[-2]}")
    
    # Pip packages
    pip_packages = [
        "pip install tqdm rich typer",
        "pip install plotly pyvista tensorboard wandb",
        "pip install memory-profiler albumentations",
        "pip install point-cloud-utils pymeshlab mcubes"
    ]
    
    for cmd in pip_packages:
        run_command(cmd, f"Installing: {cmd}")

def test_imports():
    """Test if key imports work"""
    print("\n" + "="*60)
    print("TESTING IMPORTS")
    print("="*60)
    
    test_modules = [
        ("tensorflow", "tf"),
        ("tqdm", "tqdm"),
        ("numpy", "np"),
        ("PIL", "PIL"),
        ("cv2", "cv2"),
        ("matplotlib.pyplot", "plt"),
        ("yaml", "yaml"),
        ("fastapi", "fastapi")
    ]
    
    results = {}
    for module, alias in test_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"[OK] {module}")
            results[module] = True
        except ImportError as e:
            print(f"[ERROR] {module} - {e}")
            results[module] = False
    
    return results

def create_simple_test():
    """Create a simple test to verify the installation"""
    test_code = '''
import sys
import os

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("Testing TensorFlow...")
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        print("Testing numpy...")
        import numpy as np
        test_array = np.random.rand(10, 10)
        print(f"Numpy test array shape: {test_array.shape}")
        
        print("Testing PIL...")
        from PIL import Image
        test_img = Image.new('RGB', (100, 100), color='red')
        print(f"PIL test image mode: {test_img.mode}")
        
        print("Testing tqdm...")
        from tqdm import tqdm
        import time
        for i in tqdm(range(10), desc="Testing tqdm"):
            time.sleep(0.01)
        
        print("\\n[SUCCESS] All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\\n[ERROR] Test failed: {e}")
        return False

if __name__ == "__main__":
    test_basic_functionality()
'''
    
    with open("simple_test.py", "w") as f:
        f.write(test_code)
    
    print("\n[INFO] Created simple_test.py")
    print("[INFO] Run with: python simple_test.py")

def main():
    print("FASHION 2D-to-3D GAN - QUICK INSTALLATION FIX")
    print("="*60)
    
    # Install dependencies
    install_core_dependencies()
    
    # Test imports
    results = test_imports()
    
    # Create simple test
    create_simple_test()
    
    # Summary
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    
    failed = [module for module, success in results.items() if not success]
    if failed:
        print(f"[WARNING] The following modules still need attention: {', '.join(failed)}")
        print("\nTry installing them manually:")
        for module in failed:
            if module == "tensorflow":
                print(f"  conda install tensorflow=2.13.0")
            elif module == "cv2":
                print(f"  conda install opencv")
            else:
                print(f"  conda install {module}")
    else:
        print("[SUCCESS] All core modules installed successfully!")
    
    print("\nNext steps:")
    print("1. Run: python simple_test.py")
    print("2. Run: python deploy.py --test")
    print("3. If tests pass, run: python deploy.py --analyze")

if __name__ == "__main__":
    main()
