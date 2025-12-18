#!/usr/bin/env python3
"""
Test script to find available camera sources and test streaming URLs
"""

import cv2
import sys

print("=" * 60)
print("📹 CAMERA SOURCE TESTER")
print("=" * 60)

# Test local camera devices
print("\n🔍 Testing local camera devices (0-9)...")
print("-" * 60)
available_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            print(f"✅ Camera {i}: Available ({w}x{h})")
            available_cameras.append(i)
        else:
            print(f"⚠️  Camera {i}: Opens but can't read frame")
        cap.release()
    else:
        print(f"❌ Camera {i}: Not available")

print(f"\n📊 Found {len(available_cameras)} working camera(s): {available_cameras}")

# Test network URLs
print("\n" + "=" * 60)
print("🌐 Testing Network Streaming URLs")
print("=" * 60)

# Get URL from user
print("\nEnter your phone's streaming URL (or press Enter to skip):")
print("Examples:")
print("  http://192.168.1.100:8080/video")
print("  http://10.37.6.46:4747/mjpegfeed")
url = input("\nURL: ").strip()

if url:
    print(f"\n🔗 Testing: {url}")
    
    # Try alternative formats
    test_urls = [
        url,
        url.replace("/video", "/mjpegfeed"),
        url.replace("/mjpegfeed", "/video"),
        f"{url.rsplit('/', 1)[0]}/mjpegfeed",
        f"{url.rsplit('/', 1)[0]}/video",
    ]
    
    # Remove duplicates
    test_urls = list(dict.fromkeys(test_urls))
    
    for test_url in test_urls:
        print(f"\nTrying: {test_url}")
        cap = cv2.VideoCapture(test_url)
        
        if cap.isOpened():
            print("  ⏳ Connection opened, reading frame...")
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✅ SUCCESS! Resolution: {w}x{h}")
                print(f"\n  🎯 USE THIS URL IN GUI: {test_url}")
                cap.release()
                break
            else:
                print("  ❌ Can't read frame from stream")
        else:
            print("  ❌ Can't open stream")
        
        cap.release()

print("\n" + "=" * 60)
print("✅ Test Complete!")
print("=" * 60)
print("\n💡 Next Steps:")
if available_cameras:
    print(f"   For local camera, use: {available_cameras[0]}")
print("   For phone camera, use the URL marked '🎯 USE THIS' above")
print("   Enter it in the Camera Source field in the GUI")
print("=" * 60)
