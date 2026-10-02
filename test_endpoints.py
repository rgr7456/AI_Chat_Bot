#!/usr/bin/env python3
"""
Simple test script to help debug 400 Bad Request errors.
"""

import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://127.0.0.1:8000"

def test_basic_endpoints():
    """Test basic endpoints that don't require files"""
    print("=== Testing Basic Endpoints ===")
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"GET /: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"GET /: ERROR - {e}")
    
    # Test collections list
    try:
        response = requests.get(f"{BASE_URL}/face/milvus/collections")
        print(f"GET /face/milvus/collections: {response.status_code}")
        if response.status_code == 200:
            print(f"  Collections: {response.json()}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"GET /face/milvus/collections: ERROR - {e}")
    
    # Test staff list
    try:
        response = requests.get(f"{BASE_URL}/face/milvus/staff_list")
        print(f"GET /face/milvus/staff_list: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Staff count: {len(data.get('staff', []))}")
        else:
            print(f"  Error: {response.text}")
    except Exception as e:
        print(f"GET /face/milvus/staff_list: ERROR - {e}")

def test_register_endpoint():
    """Test the register endpoint with a simple image"""
    print("\n=== Testing Register Endpoint ===")
    
    # Create a simple test image (1x1 pixel JPEG)
    import io
    from PIL import Image
    
    # Create minimal test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test with correct multipart/form-data format
    files = {
        'image': ('test.jpg', img_bytes.getvalue(), 'image/jpeg')
    }
    data = {
        'staff_id': 'test_001',
        'staff_name': 'Test User'
    }
    
    try:
        response = requests.post(f"{BASE_URL}/face/register", files=files, data=data)
        print(f"POST /face/register: {response.status_code}")
        print(f"  Response: {response.text}")
        
        if response.status_code == 400:
            print("  ❌ 400 Bad Request - Check the error details above")
        elif response.status_code == 200:
            print("  ✅ Registration successful")
        
    except Exception as e:
        print(f"POST /face/register: ERROR - {e}")

def test_verify_endpoint():
    """Test the verify endpoint"""
    print("\n=== Testing Verify Endpoint ===")
    
    # Create a simple test image
    import io
    from PIL import Image
    
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    files = {
        'image': ('verify.jpg', img_bytes.getvalue(), 'image/jpeg')
    }
    
    try:
        response = requests.post(f"{BASE_URL}/face/verify", files=files)
        print(f"POST /face/verify: {response.status_code}")
        print(f"  Response: {response.text}")
        
        if response.status_code == 400:
            print("  ❌ 400 Bad Request - Check the error details above")
        elif response.status_code == 404:
            print("  ℹ️ Staff not found (expected if no data)")
        elif response.status_code == 200:
            print("  ✅ Verification successful")
            
    except Exception as e:
        print(f"POST /face/verify: ERROR - {e}")

def show_curl_examples():
    """Show curl command examples for testing"""
    print("\n=== CURL Examples for Manual Testing ===")
    
    print("1. Test root endpoint:")
    print("   curl http://127.0.0.1:8000/")
    
    print("\n2. Test collections:")
    print("   curl http://127.0.0.1:8000/face/milvus/collections")
    
    print("\n3. Test register (replace path/to/image.jpg with actual image):")
    print('   curl -X POST "http://127.0.0.1:8000/face/register" \\')
    print('        -F "staff_id=test001" \\')
    print('        -F "staff_name=Test User" \\')
    print('        -F "image=@path/to/image.jpg"')
    
    print("\n4. Test verify:")
    print('   curl -X POST "http://127.0.0.1:8000/face/verify" \\')
    print('        -F "image=@path/to/image.jpg"')

if __name__ == "__main__":
    print("FastAPI Endpoint Tester")
    print("======================")
    
    # Test basic endpoints first
    test_basic_endpoints()
    
    # Test file upload endpoints
    test_register_endpoint()
    test_verify_endpoint()
    
    # Show manual testing examples
    show_curl_examples()
    
    print("\n=== Debugging Tips ===")
    print("1. Make sure FastAPI server is running on http://127.0.0.1:8000")
    print("2. Check server logs for detailed error messages")
    print("3. Use browser to visit http://127.0.0.1:8000/docs for Swagger UI")
    print("4. Verify image files are valid JPEG/PNG format")
    print("5. Ensure multipart/form-data is used for file uploads")