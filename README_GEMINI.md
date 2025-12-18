# 🤖 Gemini-Based Visual Inspection Pipeline

> **Replace LLaVA with Google's Gemini 2.5 Flash for automated visual inspection**

## ⚡ TL;DR - Get Running in 60 Seconds

```bash
cd /home/rishan/Desktop/LLaVA
./setup_gemini_env.sh                              # Install everything
export GEMINI_API_KEY='your-api-key-here'          # Set API key
./run_gemini.sh                                    # Run inspection
```

Done! Check `gemini_results/` for outputs.

---

## 🎯 What This Does

Automated visual inspection pipeline that:
1. **Detects** electrical sockets, fire safety equipment, and lighting using YOLO-World
2. **Inspects** each detected item using Gemini 2.5 Flash Vision AI
3. **Reports** condition as OK or FAULTY with reasoning

---

## 📋 Prerequisites

- Ubuntu 20.04+ (or similar Linux)
- Sudo access (for installing Python 3.9)
- Internet connection
- Gemini API key (free from [Google AI Studio](https://makersuite.google.com/app/apikey))

---

## 🚀 Installation

### Automated (Recommended)

```bash
./setup_gemini_env.sh
```

This installs:
- ✅ Python 3.9
- ✅ Virtual environment
- ✅ PyTorch (CPU version)
- ✅ Ultralytics YOLO
- ✅ Google Gemini SDK
- ✅ OpenCV and other dependencies

### Manual (If automated fails)

See `SETUP_INSTRUCTIONS.md` for step-by-step manual installation.

---

## 🔑 API Key Setup

### Get Your Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key

### Set the Key

**Option 1: Temporary (this session only)**
```bash
export GEMINI_API_KEY='your-key-here'
```

**Option 2: Permanent (recommended)**
```bash
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Verify it's set:**
```bash
echo $GEMINI_API_KEY
```

---

## 🎮 Usage

### Quick Run

```bash
./run_gemini.sh
```

### Manual Run

```bash
source venv/bin/activate
python main_gemini.py
deactivate
```

### Test Installation

```bash
./test_setup.sh
```

---

## 📤 Output

### Console Output

```
🔍 Starting automated inspection with Gemini 2.5 Flash...
✅ GPU detected: NVIDIA GeForce RTX 3080
📋 Scanning for all elements: wall socket, power outlet, fire extinguisher...

🚗 Moving to inspect sockets (detected as: wall socket, confidence: 0.85)...
📍 Position reached. Starting detailed inspection...
🤖 Inspecting sockets up close with Gemini 2.5 Flash...
  📸 Cropped image saved to: gemini_results/cropped_sockets_example_1.jpg
🔎 Inspection result: OK: wall socket in good condition
✅ sockets marked as INSPECTED
```

### File Outputs

```
gemini_results/
├── annotated_example_1.jpg          # Full image with bounding boxes
├── cropped_sockets_example_1.jpg    # Close-up of socket
├── cropped_lighting_example_1.jpg   # Close-up of light
└── ...
```

---

## ⚙️ Configuration

### Change Detection Model

```python
# In main_gemini.py, line 221
model = YOLO("yolov8l-world.pt")  # Options: yolov8s, yolov8m, yolov8l, yolov8x
```

### Adjust Confidence Threshold

```python
# In main_gemini.py, line 143
results = model.predict(source=image_path, conf=0.10, ...)
# Lower = more detections, Higher = fewer false positives
```

### Add Custom Objects

```python
# In main_gemini.py, lines 15-19
INSPECTION_ELEMENTS = {
    "sockets": ["wall socket", "power outlet"],
    "your_category": ["object1", "object2", "synonym"],
}
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY not set` | Run: `export GEMINI_API_KEY='your-key'` |
| `venv not found` | Run: `./setup_gemini_env.sh` |
| `No module named 'google.generativeai'` | Run: `pip install google-generativeai` |
| `Image not found` | Ensure `example_1.jpg` exists or update path in code |
| `Permission denied` | Run: `chmod +x *.sh` |

---

## 📚 Architecture

```
┌─────────────────┐
│   Input Image   │
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│  YOLO-World Detection   │
│  (Local GPU/CPU)        │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  Crop Detected Objects  │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  Gemini 2.5 Flash VLM   │
│  (Cloud API)            │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  Inspection Report      │
│  (OK/FAULTY + Reason)   │
└─────────────────────────┘
```

---

## 🎓 How It Works

1. **Detection Phase**
   - YOLO-World scans the image for target objects
   - Uses synonym-aware prompting (e.g., "wall socket" OR "power outlet")
   - Creates bounding boxes around detections

2. **Cropping Phase**
   - Extracts each detected object with padding
   - Saves crops for visual inspection
   - Maintains reference to original image

3. **Inspection Phase**
   - Sends cropped image to Gemini 2.5 Flash via REST API
   - Gemini analyzes object condition
   - Returns structured response: "OK" or "FAULTY: [reason]"

4. **Reporting Phase**
   - Logs results to console
   - Saves annotated images
   - Categorizes findings by element type

---

## 🔬 Advanced Usage

### Process Multiple Images

```python
import glob
for img in glob.glob("images/*.jpg"):
    # Call detection and inspection functions
```

### Export Results to JSON

```python
import json
results = []
# ... run inspections, collect results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### Compare with Standard YOLO

```bash
python evaluate_models_comparison.py
```

---

## 💰 Costs

- **Gemini 2.0 Flash Exp**: Currently free (verify at [Google AI pricing](https://ai.google.dev/pricing))
- **Compute**: Local detection uses your GPU/CPU (no cloud costs)
- **Storage**: Results saved locally (no storage fees)

---

## 🆚 Gemini vs LLaVA Comparison

| Feature | Gemini (this script) | LLaVA (main.py) |
|---------|---------------------|-----------------|
| Setup Complexity | ⭐ Easy | ⭐⭐⭐ Complex |
| API Key Required | ✅ Yes (free tier) | ❌ No |
| Local Inference | ❌ Detection only | ✅ Fully local |
| GPU Required | 🟡 Optional (for speed) | ✅ Yes (for VLM) |
| Internet Required | ✅ Yes | ❌ No |
| Response Speed | ⚡ Fast (API) | 🐢 Slower (local) |
| Privacy | 🟡 Cloud-based | ✅ Fully private |

---

## 📊 Model Files

- `yolov8l-world.pt` - YOLO-World large (detection)
- `yolov8x-world.pt` - YOLO-World extra large (better accuracy)
- `yolov8n_power_switch.pt` - Custom trained socket detector

---

## 🎯 Use Cases

- ✅ Electrical safety inspections
- ✅ Fire safety compliance checks
- ✅ Building maintenance audits
- ✅ Quality control automation
- ✅ Defect detection workflows

---

## 🔗 Related Scripts

- `evaluate_models_comparison.py` - Benchmark YOLO models
- `zero_shot_inference.py` - Zero-shot detection examples
- `main.py` - LLaVA-based inspection (local inference)

---

## 📞 Support

For issues or questions:
1. Run `./test_setup.sh` to diagnose
2. Check error messages in console
3. Review troubleshooting section
4. Verify API key is set: `echo $GEMINI_API_KEY`

---

## 🎉 Next Steps

Once you have it running:

1. ✅ Test with your own images
2. ✅ Customize object categories
3. ✅ Adjust confidence thresholds
4. ✅ Integrate into your workflow
5. ✅ Export results to your preferred format

---

**Ready to start? Run `./run_gemini.sh` and let's inspect! 🚀**

*For detailed documentation, see `QUICKSTART.md` or `GET_STARTED.md`*
