# 🎯 Complete Setup & Usage Summary

## 📦 What's Included

This workspace contains a complete visual inspection pipeline with two AI approaches:

### 1. **Gemini 2.5 Flash Pipeline** (Recommended - Cloud-based)
- Uses Google's Gemini AI for visual inspection
- REST API integration (works with Python 3.9+)
- No local GPU required for VLM inference
- Easy setup and deployment

### 2. **LLaVA Pipeline** (Advanced - Local inference)
- Uses LLaVA for visual inspection
- Requires local LLaVA server setup
- More complex but fully offline

---

## 🚀 Getting Started (Choose One)

### Option A: Super Quick Start (Recommended)

```bash
cd /home/rishan/Desktop/LLaVA

# 1. Run the automated setup (installs Python 3.9, creates venv, installs packages)
./setup_gemini_env.sh

# 2. Set your Gemini API key
export GEMINI_API_KEY='your-api-key-here'

# 3. Run the inspection
./run_gemini.sh
```

That's it! The `run_gemini.sh` script handles everything automatically.

### Option B: Manual Step-by-Step

```bash
# 1. Install Python 3.9
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# 2. Create virtual environment
cd /home/rishan/Desktop/LLaVA
python3.9 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python pillow google-generativeai requests

# 4. Set API key
export GEMINI_API_KEY='your-api-key-here'

# 5. Run inspection
python main_gemini.py
```

---

## 📋 Available Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_gemini_env.sh` | Automated environment setup | `./setup_gemini_env.sh` |
| `run_gemini.sh` | One-command launcher | `./run_gemini.sh` |
| `test_setup.sh` | Verify installation | `./test_setup.sh` |
| `main_gemini.py` | Gemini-based inspection | `python main_gemini.py` |
| `evaluate_models_comparison.py` | Model comparison | `python evaluate_models_comparison.py` |
| `main.py` | LLaVA-based inspection | `python main.py` (requires LLaVA server) |

---

## 📖 Documentation Files

- **QUICKSTART.md** - Comprehensive getting started guide
- **SETUP_INSTRUCTIONS.md** - Detailed setup instructions
- **README_ZERO_SHOT.md** - Zero-shot detection documentation
- **This file** - Quick reference summary

---

## 🎬 Typical Workflow

### First Time Setup

```bash
# 1. Clone or navigate to workspace
cd /home/rishan/Desktop/LLaVA

# 2. Run setup (one-time)
./setup_gemini_env.sh

# 3. Get Gemini API key from https://makersuite.google.com/app/apikey

# 4. Add to your environment (permanent)
echo 'export GEMINI_API_KEY="your-key-here"' >> ~/.bashrc
source ~/.bashrc

# 5. Test everything works
./test_setup.sh
```

### Daily Usage

```bash
# Quick run (handles activation automatically)
./run_gemini.sh

# OR manual run
source venv/bin/activate
python main_gemini.py
deactivate
```

---

## 🔍 What the Inspection Does

1. **Object Detection** (YOLO-World)
   - Scans image for: sockets, fire extinguishers, lighting
   - Creates bounding boxes
   - Saves annotated overview

2. **Visual Inspection** (Gemini VLM)
   - Crops each detected object
   - Analyzes with Gemini 2.5 Flash
   - Determines: OK or FAULTY
   - Provides reasoning if faulty

3. **Results Output**
   - Console: Real-time inspection log
   - `gemini_results/annotated_*.jpg` - Annotated full images
   - `gemini_results/cropped_*.jpg` - Cropped object close-ups

---

## 🛠️ Common Tasks

### Change Test Image

Edit `main_gemini.py` line 186-194 to point to your image:
```python
possible_paths = [
    "your_image.jpg",
    # ... other paths
]
```

### Add New Object Types

Edit `main_gemini.py` lines 15-19:
```python
INSPECTION_ELEMENTS = {
    "sockets": ["wall socket", "power outlet"],
    "your_category": ["synonym1", "synonym2"],
}
```

### Adjust Detection Sensitivity

Edit `main_gemini.py` line 143:
```python
results = model.predict(source=image_path, conf=0.10, ...)
# Lower conf = more detections (more false positives)
# Higher conf = fewer detections (more missed objects)
```

### Batch Process Multiple Images

Modify `main()` function to loop through a directory:
```python
import glob
for image_path in glob.glob("images/*.jpg"):
    # ... inspection code
```

---

## 🔧 Troubleshooting Quick Reference

| Problem | Solution |
|---------|----------|
| API key not found | `export GEMINI_API_KEY='your-key'` |
| venv not found | Run `./setup_gemini_env.sh` |
| Import errors | `pip install --upgrade google-generativeai` |
| Image not found | Check `example_1.jpg` exists or update path |
| GPU not detected | Normal if no NVIDIA GPU; uses CPU automatically |
| Permission denied | `chmod +x setup_gemini_env.sh run_gemini.sh test_setup.sh` |

---

## 📊 Project Structure

```
/home/rishan/Desktop/LLaVA/
├── Scripts (Ready to Run)
│   ├── run_gemini.sh          # ⭐ Main launcher
│   ├── setup_gemini_env.sh    # ⭐ One-time setup
│   ├── test_setup.sh          # Verify installation
│   ├── main_gemini.py         # Gemini inspection pipeline
│   ├── main.py                # LLaVA inspection pipeline
│   └── evaluate_models_comparison.py  # Model benchmarking
│
├── Documentation
│   ├── QUICKSTART.md          # Comprehensive guide
│   ├── SETUP_INSTRUCTIONS.md  # Detailed setup
│   └── README_ZERO_SHOT.md    # Zero-shot detection
│
├── Models
│   ├── yolov8l-world.pt       # YOLO-World (for inspection)
│   ├── yolov8n.pt             # Standard YOLO
│   └── yolov8n_power_switch.pt # Custom trained model
│
├── Test Images
│   ├── example_1.jpg
│   ├── example_2.jpg
│   └── ... (more examples)
│
├── Results
│   ├── gemini_results/        # Gemini inspection outputs
│   └── zero_shot_results/     # Zero-shot detection outputs
│
└── Training Data
    └── socket_training/       # Custom model training dataset
```

---

## 💡 Pro Tips

1. **First Time**: Always run `./test_setup.sh` after setup
2. **API Key**: Save to `~/.bashrc` for permanent availability
3. **GPU**: Script auto-detects, no configuration needed
4. **Costs**: Gemini 2.0 Flash Exp is currently free (verify current pricing)
5. **Speed**: GPU ~10x faster for detection, but VLM is cloud-based
6. **Accuracy**: Use `yolov8x-world.pt` for better detection (slower)

---

## 🎓 Learn More

- **Gemini API**: https://ai.google.dev/docs
- **YOLO-World**: https://docs.ultralytics.com/models/yolo-world/
- **Ultralytics**: https://docs.ultralytics.com/

---

## 🎯 Quick Commands Cheat Sheet

```bash
# Setup (one-time)
./setup_gemini_env.sh

# Set API key (one-time, permanent)
echo 'export GEMINI_API_KEY="your-key"' >> ~/.bashrc && source ~/.bashrc

# Run inspection (daily use)
./run_gemini.sh

# Test installation
./test_setup.sh

# Manual activation
source venv/bin/activate

# Run model comparison
source venv/bin/activate && python evaluate_models_comparison.py

# Check what's installed
source venv/bin/activate && pip list

# Update packages
source venv/bin/activate && pip install --upgrade google-generativeai ultralytics
```

---

## ✅ Success Checklist

- [ ] Python 3.9 installed
- [ ] Virtual environment created
- [ ] All packages installed (`./test_setup.sh` passes)
- [ ] Gemini API key set
- [ ] Test image available
- [ ] First inspection run successful
- [ ] Results appear in `gemini_results/`

---

**You're all set! Run `./run_gemini.sh` to start inspecting! 🚀**
