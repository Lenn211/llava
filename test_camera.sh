#!/bin/bash
# Quick test script for camera connectivity

echo "======================================"
echo "Camera Connection Test Script"
echo "======================================"
echo ""

echo "Testing available cameras..."
echo ""

# Test webcam 0
echo "1. Testing webcam 0..."
python3 -c "
import cv2
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print('   ✅ Webcam 0: Available')
    cap.release()
else:
    print('   ❌ Webcam 0: Not available')
"

# Test webcam 1
echo "2. Testing webcam 1..."
python3 -c "
import cv2
cap = cv2.VideoCapture(1)
if cap.isOpened():
    print('   ✅ Webcam 1: Available')
    cap.release()
else:
    print('   ❌ Webcam 1: Not available')
"

echo ""
echo "======================================"
echo "Phone Camera Setup Instructions"
echo "======================================"
echo ""
echo "To use your phone camera:"
echo "1. Install 'IP Webcam' app on Android"
echo "2. Start the server in the app"
echo "3. Note the IP address (e.g., 192.168.1.100:8080)"
echo "4. In the GUI, enter: http://YOUR_IP:8080/video"
echo ""
echo "Example URLs:"
echo "  - Webcam: 0"
echo "  - Phone:  http://192.168.1.100:8080/video"
echo ""

echo "======================================"
echo "Ready to start!"
echo "Run: python main_gemini_gui_camera.py"
echo "======================================"
