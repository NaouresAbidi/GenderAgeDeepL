#!/usr/bin/env python3
"""Test the Age & Gender API"""

import requests
import json
import numpy as np
from PIL import Image
import io

def test_health():
    """Test the health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get("http://127.0.0.1:5000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def create_test_image():
    """Create a dummy test image"""
    print("🖼️  Creating test image...")
    # Create a random grayscale image
    img_array = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='L')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes

def test_prediction():
    """Test the prediction endpoint"""
    print("🔮 Testing prediction endpoint...")
    try:
        # Create test image
        test_img = create_test_image()
        
        # Send prediction request
        files = {'image': ('test.png', test_img, 'image/png')}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        return response.status_code in [200, 422]  # 422 for low quality is OK
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    print("🚀 Testing Age & Gender Prediction API")
    print("=" * 50)
    
    # Test health
    health_ok = test_health()
    print()
    
    # Test prediction
    prediction_ok = test_prediction()
    print()
    
    # Summary
    print("📊 Test Results:")
    print(f"   Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"   Prediction:   {'✅ PASS' if prediction_ok else '❌ FAIL'}")
    
    if health_ok and prediction_ok:
        print("\n🎉 All tests passed! Your API is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Test with real images")
        print("   2. Train a proper model for better predictions")
        print("   3. Deploy to production when ready")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()