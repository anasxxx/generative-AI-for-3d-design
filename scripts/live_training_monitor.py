#!/usr/bin/env python3
"""
Live Training Progress Monitor
Shows real-time training progress with detailed metrics
"""

import os
import time
import json
import psutil
from pathlib import Path
from datetime import datetime, timedelta
import threading
import sys

class LiveTrainingMonitor:
    def __init__(self):
        self.start_time = None
        self.last_checkpoint_time = None
        self.training_process = None
        
    def find_training_process(self):
        """Find the training process"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] == 'python.exe':
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'train.py' in cmdline:
                        return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None
    
    def get_process_stats(self, proc):
        """Get process statistics"""
        try:
            cpu_percent = proc.cpu_percent()
            memory_info = proc.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            create_time = datetime.fromtimestamp(proc.create_time())
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'create_time': create_time,
                'runtime': datetime.now() - create_time
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def get_latest_checkpoint(self):
        """Get the latest checkpoint file"""
        checkpoint_dir = Path('./models/checkpoints/')
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob('*.h5'))
            if checkpoints:
                latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
                mtime = datetime.fromtimestamp(latest.stat().st_mtime)
                return latest.name, mtime
        return None, None
    
    def get_training_data_info(self):
        """Get training data information"""
        train_dir = Path('./data/train')
        if train_dir.exists():
            train_images = len(list((train_dir / 'images').glob('*.npy')))
            train_voxels = len(list((train_dir / 'voxels').glob('*.npy')))
            return train_images, train_voxels
        return 0, 0
    
    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_time(self, seconds):
        """Format time in HH:MM:SS"""
        return str(timedelta(seconds=int(seconds)))
    
    def display_progress(self):
        """Display live training progress"""
        while True:
            self.clear_screen()
            
            print("🚀 FASHION 2D-to-3D GAN - LIVE TRAINING MONITOR")
            print("=" * 60)
            print(f"⏰ Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check training process
            proc = self.find_training_process()
            if proc:
                stats = self.get_process_stats(proc)
                if stats:
                    print(f"✅ Training Process: PID {proc.pid}")
                    print(f"🖥️  CPU Usage: {stats['cpu_percent']:.1f}%")
                    print(f"💾 Memory Usage: {stats['memory_mb']:.1f} MB")
                    print(f"⏱️  Runtime: {self.format_time(stats['runtime'].total_seconds())}")
                    print()
                    
                    # Calculate progress based on runtime
                    if self.start_time is None:
                        self.start_time = stats['create_time']
                    
                    elapsed = datetime.now() - self.start_time
                    target_duration = timedelta(hours=8)
                    progress_percent = min(100, (elapsed.total_seconds() / target_duration.total_seconds()) * 100)
                    
                    print(f"📊 Training Progress: {progress_percent:.1f}%")
                    print(f"⏳ Elapsed: {self.format_time(elapsed.total_seconds())}")
                    print(f"🎯 Target: 8:00:00")
                    print(f"⏰ Remaining: {self.format_time(max(0, target_duration.total_seconds() - elapsed.total_seconds()))}")
                    print()
                else:
                    print("❌ Training process found but cannot access stats")
            else:
                print("❌ No training process found")
                print("💡 Start training with: python scripts/train.py --hours 8")
                print()
            
            # Check checkpoints
            checkpoint_name, checkpoint_time = self.get_latest_checkpoint()
            if checkpoint_name and checkpoint_time:
                print(f"💾 Latest Checkpoint: {checkpoint_name}")
                print(f"📅 Saved: {checkpoint_time.strftime('%H:%M:%S')}")
                
                if self.last_checkpoint_time != checkpoint_time:
                    print("🆕 NEW CHECKPOINT SAVED!")
                    self.last_checkpoint_time = checkpoint_time
                print()
            else:
                print("⏳ No checkpoints yet")
                print()
            
            # Training data info
            train_images, train_voxels = self.get_training_data_info()
            print(f"📊 Training Data: {train_images} images, {train_voxels} voxels")
            
            # Model info
            print(f"🤖 Model: Fashion3DGAN with ResNet50 pretrained encoder")
            print(f"🎯 Goal: Fine-tune on real fashion dataset")
            print()
            
            # Status indicators
            if proc and stats and stats['cpu_percent'] > 0:
                print("🟢 TRAINING IS ACTIVE")
                print("   • Model is learning from real fashion data")
                print("   • ResNet50 encoder is being fine-tuned")
                print("   • 3D voxel generation is improving")
            else:
                print("🔴 TRAINING IS NOT ACTIVE")
                print("   • Check if training process is running")
                print("   • Verify data and model setup")
            
            print()
            print("=" * 60)
            print("Press Ctrl+C to stop monitoring")
            
            # Wait before next update
            time.sleep(2)
    
    def start_monitoring(self):
        """Start the live monitoring"""
        try:
            print("🔍 Starting live training monitor...")
            print("📡 Monitoring training progress every 2 seconds...")
            print("⏳ Initializing...")
            time.sleep(3)
            
            self.display_progress()
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
            print("📊 Training may still be running in background")

if __name__ == "__main__":
    monitor = LiveTrainingMonitor()
    monitor.start_monitoring()
