#!/usr/bin/env python3
"""
Monitor training progress
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime

def monitor_training():
    """Monitor training progress"""
    print("🔍 Training Monitor")
    print("=" * 40)
    
    # Check for training processes
    import psutil
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python.exe' and 'train.py' in ' '.join(proc.info['cmdline'] or []):
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_processes:
        print(f"✅ Training process running: PID {python_processes[0]['pid']}")
    else:
        print("❌ No training process found")
        return
    
    # Check for checkpoints
    checkpoint_dir = Path('./models/checkpoints/')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.h5'))
        if checkpoints:
            print(f"✅ Found {len(checkpoints)} checkpoints")
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"📁 Latest: {latest.name}")
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
    
    print(f"⏰ Current time: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 40)

if __name__ == "__main__":
    while True:
        monitor_training()
        print("\n🔄 Checking again in 30 seconds...")
        time.sleep(30)
