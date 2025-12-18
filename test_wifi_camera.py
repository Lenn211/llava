#!/usr/bin/env python3
"""
WiFi Camera Connection Diagnostic Tool
Tests various WiFi camera connection methods and URLs
"""

import cv2
import time
import sys

def test_camera_connection(url, description):
    """Test a camera connection with detailed diagnostics"""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    print(f"{'='*70}")
    
    try:
        # Try to open the camera
        print("Opening camera...")
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            print("❌ FAILED: Camera did not open")
            return False
        
        print("✓ Camera opened successfully")
        
        # Try to read a frame
        print("Reading frame...")
        ret, frame = cap.read()
        
        if not ret:
            print("❌ FAILED: Could not read frame")
            cap.release()
            return False
        
        print(f"✓ Frame read successfully - Shape: {frame.shape}")
        
        # Try reading multiple frames
        print("Testing continuous frame reading (5 frames)...")
        success_count = 0
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                success_count += 1
                print(f"  Frame {i+1}: ✓ ({frame.shape})")
            else:
                print(f"  Frame {i+1}: ❌ Failed")
            time.sleep(0.1)
        
        cap.release()
        
        if success_count >= 3:
            print(f"✅ SUCCESS: {success_count}/5 frames read successfully")
            print(f"👍 This URL works! Use: {url}")
            return True
        else:
            print(f"⚠️  PARTIAL: Only {success_count}/5 frames read")
            return False
            
    except Exception as e:
        print(f"❌ EXCEPTION: {type(e).__name__}: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         WiFi Camera Connection Diagnostic Tool                   ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Get IP address from user
    if len(sys.argv) > 1:
        ip = sys.argv[1]
    else:
        ip = input("Enter your phone's IP address (e.g., 192.168.1.100): ").strip()
    
    if not ip:
        print("No IP address provided!")
        return
    
    print(f"\nTesting WiFi camera connections for IP: {ip}")
    
    # Test different URL formats
    test_urls = [
        # iVCam
        (f"http://{ip}:8080/video", "iVCam - Default"),
        (f"http://{ip}:8080/", "iVCam - Root"),
        
        # DroidCam
        (f"http://{ip}:4747/video", "DroidCam - Default"),
        (f"http://{ip}:4747/mjpegfeed", "DroidCam - MJPEG"),
        
        # IP Webcam
        (f"http://{ip}:8080/video", "IP Webcam - Video"),
        (f"http://{ip}:8080/videofeed", "IP Webcam - Video Feed"),
        (f"http://{ip}:8080/shot.jpg", "IP Webcam - JPEG Stream"),
        
        # EpocCam
        (f"http://{ip}:8080/video", "EpocCam - Default"),
        
        # Generic RTSP (some apps support this)
        (f"rtsp://{ip}:8554/video", "RTSP Stream - 8554"),
        (f"rtsp://{ip}:8080/", "RTSP Stream - 8080"),
        
        # Generic HTTP MJPEG
        (f"http://{ip}:8081/video", "HTTP MJPEG - Port 8081"),
    ]
    
    successful_urls = []
    
    for url, description in test_urls:
        if test_camera_connection(url, description):
            successful_urls.append((url, description))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if successful_urls:
        print(f"\n✅ Found {len(successful_urls)} working connection(s):\n")
        for url, description in successful_urls:
            print(f"  {description}")
            print(f"  → {url}\n")
        
        print("\n📝 To use in the GUI:")
        print(f"   Enter this URL in the 'Camera Source' field:")
        print(f"   {successful_urls[0][0]}")
    else:
        print("\n❌ No working connections found!")
        print("\nTroubleshooting steps:")
        print("1. Check that your phone and computer are on the same WiFi network")
        print("2. Verify the IP address is correct (check in the camera app)")
        print("3. Make sure the camera app is running on your phone")
        print("4. Check if your firewall is blocking the connection")
        print("5. Try disabling WiFi isolation/AP isolation in your router settings")
        print("6. Some apps may use different ports - check the app's settings")
        
        print("\n🔍 Alternative test:")
        print(f"   Try opening in a web browser: http://{ip}:8080/video")
        print("   If you can see the video feed in the browser, the URL should work")

if __name__ == "__main__":
    main()
