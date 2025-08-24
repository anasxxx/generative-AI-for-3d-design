#!/usr/bin/env python3
"""
Check training progress directly
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def check_training():
    """Check training progress"""
    print("🔍 Training Progress Check")
    print("=" * 50)
    
    # Check Python processes
    import psutil
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent']):
        try:
            if proc.info['name'] == 'python.exe':
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'train.py' in cmdline:
                    python_processes.append({
                        'pid': proc.info['pid'],
                        'cpu': proc.info['cpu_percent'],
                        'cmdline': cmdline
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print(f"✅ Found {len(python_processes)} training processes:")
        for proc in python_processes:
            print(f"   PID {proc['pid']}: CPU {proc['cpu']:.1f}%")
    else:
        print("❌ No training processes found")
    
    # Check for checkpoints
    checkpoint_dir = Path('./models/checkpoints/')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.h5'))
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} checkpoints:")
            for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime):
                mtime = datetime.fromtimestamp(cp.stat().st_mtime)
                print(f"   {cp.name} - {mtime.strftime('%H:%M:%S')}")
        else:
            print("⏳ No checkpoints yet")
    
    # Check for logs
    logs_dir = Path('./logs/')
    if logs_dir.exists():
        log_files = list(logs_dir.glob('*.log'))
        if log_files:
            print(f"📝 Found {len(log_files)} log files")
        else:
            print("📝 No log files yet")
    
    # Check training data
    train_dir = Path('./data/train')
    if train_dir.exists():
        train_images = len(list((train_dir / 'images').glob('*.npy')))
        train_voxels = len(list((train_dir / 'voxels').glob('*.npy')))
        print(f"📊 Training data: {train_images} images, {train_voxels} voxels")
    
    # Check outputs
    outputs_dir = Path('./outputs/')
    if outputs_dir.exists():
        output_files = list(outputs_dir.glob('*'))
        if output_files:
            print(f"📁 Found {len(output_files)} output files")
        else:
            print("📁 No output files yet")
    
    print(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50)

if __name__ == "__main__":
    check_training()
