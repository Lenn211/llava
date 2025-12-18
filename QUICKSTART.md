# 🚀 Quick Start Guide - Gemini-Based Inspection

This guide will help you set up and run the Gemini-based visual inspection pipeline in just a few steps.

## 🎯 What You'll Get

- **Gemini 2.5 Flash VLM Integration**: Replace LLaVA with Google's Gemini AI for visual inspection
- **YOLO Object Detection**: Detect electrical components, safety equipment, and lighting
- **Automated Environment Setup**: One-command setup script
- **Model Comparison Tools**: Compare custom vs standard YOLO models

## ⚡ Quick Setup (3 Steps)

### Step 1: Run the Setup Script

```bash
cd /home/rishan/Desktop/LLaVA
./setup_gemini_env.sh
```

This will:
- Install Python 3.9
- Create a fresh virtual environment
- Install all dependencies (PyTorch, YOLO, Gemini SDK, OpenCV)
- Verify the installation

**Note**: You'll need sudo access. The script will prompt for your password.

### Step 2: Set Your Gemini API Key

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey), then:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Or for permanent setup:

```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Run the Inspection

```bash
source venv/bin/activate
python main_gemini.py
```

## 🧪 Verify Your Setup

Test everything is working:

```bash
./test_setup.sh
```

This will check:
- ✅ Python version (should be 3.9+)
- ✅ All packages installed correctly
- ✅ API key is set
- ✅ Test images are available

## 📁 Available Scripts

### 1. **main_gemini.py** - Gemini-Based Inspection
```bash
python main_gemini.py
```

**What it does:**
- Detects electrical sockets, fire safety equipment, and lighting
- Crops detected objects for detailed inspection
- Uses Gemini 2.5 Flash to assess condition (OK/FAULTY)
- Saves annotated images and cropped objects to `gemini_results/`

**Key Features:**
- REST API implementation (works with Python 3.9+)
- Automatic GPU detection and usage
- Synonym-aware detection (e.g., "wall socket" = "power outlet")
- Structured inspection reports

### 2. **evaluate_models_comparison.py** - Model Performance Analysis
```bash
python evaluate_models_comparison.py
```

**What it does:**
- Compares YOLOv8x-world (custom) vs YOLOv8x (standard) models
- Tests on multiple datasets with synonym matching
- Reports accuracy and F1-scores per dataset
- Provides average performance metrics

### 3. **main.py** - Original LLaVA Pipeline
```bash
python main.py
```

**Note:** Requires a running LLaVA server (more complex setup)

## 🛠️ Troubleshooting

### "GEMINI_API_KEY not found"
```bash
# Check if it's set
echo $GEMINI_API_KEY

# If empty, set it
export GEMINI_API_KEY='your-key-here'
```

### "No module named 'google.generativeai'"
```bash
source venv/bin/activate
pip install --upgrade google-generativeai
```

### "Image not found"
Make sure you have `example_1.jpg` in the LLaVA directory:
```bash
ls example_*.jpg
```

### GPU Not Detected
This is normal if you don't have an NVIDIA GPU. The script will automatically use CPU (slower but functional).

### Package Installation Fails
Try installing packages individually:
```bash
source venv/bin/activate
pip install torch
pip install ultralytics
pip install google-generativeai
pip install opencv-python
```

## 📊 Output Examples

### Detection Output
```
🔍 Starting automated inspection with Gemini 2.5 Flash...
✅ GPU detected: NVIDIA GeForce RTX 3080
📋 Scanning for all elements: wall socket, power outlet, fire extinguisher...
Saved annotated image to: gemini_results/annotated_example_1.jpg
```

### Inspection Results
```
🚗 Moving to inspect sockets (detected as: wall socket, confidence: 0.85)...
📍 Position reached. Starting detailed inspection...
🤖 Inspecting sockets up close with Gemini 2.5 Flash...
  📸 Cropped image saved to: gemini_results/cropped_sockets_example_1.jpg
🔎 Inspection result: OK: wall socket in good condition
✅ sockets marked as INSPECTED
```

## 🔧 Configuration

### Change Detection Model
In `main_gemini.py`, line 221:
```python
model = YOLO("yolov8l-world.pt")  # Change to yolov8x-world.pt for more accuracy
```

### Adjust Detection Confidence
In `main_gemini.py`, line 143:
```python
results = model.predict(source=image_path, conf=0.10, verbose=False, device=device)
# Increase conf to 0.25 for stricter detection
```

### Add More Elements to Inspect
In `main_gemini.py`, lines 15-19:
```python
INSPECTION_ELEMENTS = {
    "sockets": ["wall socket", "power outlet", "electrical socket"],
    "fire_safety": ["fire extinguisher", "fire safety equipment"],
    "lighting": ["light fixture", "ceiling light", "lamp"],
    # Add your own:
    "new_category": ["synonym1", "synonym2", "synonym3"]
}
```

### Change Gemini Model
In `main_gemini.py`, line 92:
```python
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
# Change to: gemini-1.5-pro or gemini-1.5-flash
```

## 📚 Additional Resources

- **Gemini API Documentation**: https://ai.google.dev/docs
- **Ultralytics YOLO**: https://docs.ultralytics.com/
- **Model Weights**: Located in project root
  - `yolov8l-world.pt` - YOLO-World large model
  - `yolov8n_power_switch.pt` - Custom trained socket detector

## 🎓 Understanding the Pipeline

1. **Detection Phase** (YOLO-World)
   - Scans image for all specified elements
   - Creates bounding boxes around detections
   - Saves annotated overview image

2. **Inspection Phase** (Gemini VLM)
   - Crops each detected object
   - Sends crop to Gemini 2.5 Flash
   - Gets condition assessment (OK/FAULTY)
   - Reports findings

3. **Results**
   - Annotated full images in `gemini_results/`
   - Cropped objects in `gemini_results/cropped_*`
   - Console output with inspection decisions

## 💡 Tips

- **Batch Processing**: Modify `main_gemini.py` to loop through multiple images
- **Better Accuracy**: Use `yolov8x-world.pt` instead of `yolov8l-world.pt`
- **Cost Optimization**: Gemini 2.0 Flash Exp is currently free (check current pricing)
- **Speed**: GPU detection is ~10x faster than CPU

## 🆘 Getting Help

If you encounter issues:

1. Run the test script: `./test_setup.sh`
2. Check the error message carefully
3. Refer to the troubleshooting section above
4. Check package versions: `pip list`
5. Verify Python version: `python --version` (should be 3.9+)

## ✨ What's Next?

Once you have the basic pipeline running, you can:

- Test on your own images
- Fine-tune detection thresholds
- Add custom object categories
- Export results to JSON/CSV
- Integrate with a web dashboard
- Run batch processing on multiple images

---

**Happy Inspecting! 🔍🤖**
