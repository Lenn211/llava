#!/bin/bash
# Quick test script to verify Gemini setup

echo "========================================="
echo "Testing Gemini Environment Setup"
echo "========================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run ./setup_gemini_env.sh first"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Test Python version
echo ""
echo "Python version:"
python --version

# Test imports
echo ""
echo "Testing package imports..."
python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError as e:
    print(f'❌ PyTorch: {e}')

try:
    import ultralytics
    print(f'✅ Ultralytics: {ultralytics.__version__}')
except ImportError as e:
    print(f'❌ Ultralytics: {e}')

try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except ImportError as e:
    print(f'❌ OpenCV: {e}')

try:
    import google.generativeai as genai
    print(f'✅ Google GenerativeAI: Installed')
except ImportError as e:
    print(f'❌ Google GenerativeAI: {e}')

try:
    import requests
    print(f'✅ Requests: Installed')
except ImportError as e:
    print(f'❌ Requests: {e}')
"

# Check API key
echo ""
echo "API Key status:"
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set"
    echo "   Set it with: export GEMINI_API_KEY='your-api-key'"
else
    echo "✅ GEMINI_API_KEY is set"
fi

# Check for test images
echo ""
echo "Test images:"
if [ -f "example_1.jpg" ]; then
    echo "✅ example_1.jpg found"
else
    echo "⚠️  example_1.jpg not found"
fi

echo ""
echo "========================================="
echo "Test complete!"
echo "========================================="

deactivate
