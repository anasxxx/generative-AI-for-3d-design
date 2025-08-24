#!/usr/bin/env python3
"""
Test script for Fashion 2D-to-3D GAN API
Run this after the API server is started
"""

import requests
import json
import time
import numpy as np
from PIL import Image
import os

def test_api():
    """Test the API endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Fashion 2D-to-3D GAN API")
    print("=" * 50)
    
    # Test 1: Check if server is running
    print("\n1. Testing server connection...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
            data = response.json()
            print(f"   API Version: {data['version']}")
            print(f"   Status: {data['status']}")
            print(f"   Dataset: {data['dataset']}")
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("   Make sure the API server is running:")
        print("   Run: python deploy.py --api")
        return False
    
    # Test 2: Check model status
    print("\n2. Checking model status...")
    try:
        response = requests.get(f"{base_url}/model-status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Model Status:")
            print(f"   Model loaded: {status['model_loaded']}")
            print(f"   GPU available: {status['gpu_available']}")
            print(f"   Last loaded: {status['last_loaded']}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    # Test 3: Generate a test image
    print("\n3. Creating test fashion image...")
    test_image_path = create_test_fashion_image()
    
    if test_image_path:
        print(f"✅ Test image created: {test_image_path}")
        
        # Test 4: Generate 3D model
        print("\n4. Testing 3D generation...")
        try:
            with open(test_image_path, 'rb') as f:
                files = {'file': ('test_fashion.jpg', f, 'image/jpeg')}
                data = {'format': 'obj', 'quality': 'high'}
                
                print("   Sending image to API...")
                start_time = time.time()
                response = requests.post(f"{base_url}/generate", files=files, data=data, timeout=60)
                generation_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ 3D Generation successful!")
                    print(f"   Generation ID: {result['generation_id']}")
                    print(f"   Processing time: {result['processing_time']:.2f}s")
                    print(f"   Total time: {generation_time:.2f}s")
                    print(f"   Voxel occupancy: {result['model_info']['voxel_occupancy']:.3f}")
                    print(f"   Voxel shape: {result['model_info']['voxel_shape']}")
                else:
                    print(f"❌ Generation failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
        except Exception as e:
            print(f"❌ Generation error: {e}")
    
    # Test 5: Check API statistics
    print("\n5. Checking API statistics...")
    try:
        response = requests.get(f"{base_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print("✅ API Statistics:")
            print(f"   Total generations: {stats['total_generations']}")
            print(f"   Average time: {stats['average_processing_time']:.2f}s")
        else:
            print(f"❌ Stats check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Stats error: {e}")
    
    print("\n🎉 API Test Complete!")
    print(f"🌐 Interactive docs: {base_url}/docs")
    print("🎨 Ready for fashion 2D-to-3D generation!")
    
    return True

def create_test_fashion_image():
    """Create a test fashion image"""
    try:
        # Create a simple test image that looks like a fashion item
        width, height = 256, 256
        
        # Create a simple handbag-like shape
        image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        
        # Draw a simple bag shape
        center_x, center_y = width // 2, height // 2
        
        # Bag body (ellipse)
        y, x = np.ogrid[:height, :width]
        mask = ((x - center_x) ** 2 / (80 ** 2) + (y - center_y + 20) ** 2 / (60 ** 2)) <= 1
        image[mask] = [139, 69, 19]  # Brown color
        
        # Handle
        handle_mask = ((x - center_x) ** 2 / (30 ** 2) + (y - center_y + 80) ** 2 / (15 ** 2)) <= 1
        image[handle_mask] = [101, 67, 33]  # Darker brown
        
        # Add some texture/details
        for i in range(0, height, 20):
            for j in range(0, width, 20):
                if mask[i, j]:
                    image[i:i+2, j:j+10] = [160, 82, 45]  # Lighter brown stripes
        
        # Save the image
        pil_image = Image.fromarray(image)
        test_path = "test_fashion_bag.jpg"
        pil_image.save(test_path, "JPEG", quality=95)
        
        return test_path
        
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

if __name__ == "__main__":
    success = test_api()
    
    if success:
        print("\n✨ Next steps:")
        print("1. Try uploading your own fashion images")
        print("2. Test different formats: obj, ply, npy")
        print("3. Experiment with quality settings")
        print("4. Check the interactive docs at /docs")
    else:
        print("\n🔧 Troubleshooting:")
        print("1. Make sure the API server is running")
        print("2. Check for any error messages")
        print("3. Verify all dependencies are installed")
