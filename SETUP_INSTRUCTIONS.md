# Gemini-Based Inspection Setup Instructions

## Quick Setup (Automated)

Run the automated setup script:

```bash
cd /home/rishan/Desktop/LLaVA
./setup_gemini_env.sh
```

This script will:
1. Install Python 3.9
2. Create a new virtual environment
3. Install all required packages (PyTorch, Ultralytics, Gemini SDK, etc.)
4. Verify the installation

## Manual Setup (Alternative)

If you prefer to set up manually:

### Step 1: Install Python 3.9
```bash
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev
```

### Step 2: Create Virtual Environment
```bash
cd /home/rishan/Desktop/LLaVA
python3.9 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch (CPU version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install YOLO and vision packages
pip install ultralytics opencv-python pillow

# Install Gemini SDK
pip install google-generativeai

# Install other dependencies
pip install requests python-dotenv
```

## Set Up Gemini API Key

### Option 1: Environment Variable (Temporary)
```bash
export GEMINI_API_KEY='your-api-key-here'
```

### Option 2: Add to .bashrc (Permanent)
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Option 3: Create .env file
```bash
echo 'GEMINI_API_KEY=your-api-key-here' > .env
```

Then modify `main_gemini.py` to load from .env:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Running the Scripts

### Gemini-Based Inspection
```bash
source venv/bin/activate
python main_gemini.py
```

### Model Comparison (YOLOv8x-world vs YOLOv8x)
```bash
source venv/bin/activate
python evaluate_models_comparison.py
```

## Verify Installation

Check Python version:
```bash
python --version
# Should show: Python 3.9.x
```

Check installed packages:
```bash
pip list | grep -E "(torch|ultralytics|google-generativeai|opencv)"
```

## Troubleshooting

### API Key Not Found
If you see "GEMINI_API_KEY environment variable not set":
- Ensure you've set the API key as shown above
- Restart your terminal or run `source ~/.bashrc`
- Verify with: `echo $GEMINI_API_KEY`

### CUDA/GPU Issues
The script automatically detects GPU availability. If running on CPU:
- This is normal if you don't have an NVIDIA GPU
- Performance will be slower but still functional

### Package Installation Errors
If you encounter errors during package installation:
```bash
# Try installing packages one by one
pip install torch
pip install ultralytics
pip install google-generativeai
```

### ImportError for google.generativeai
If you get import errors:
```bash
pip install --upgrade google-generativeai
```

## Files Overview

- `main_gemini.py` - Gemini 2.5 Flash-based inspection pipeline
- `evaluate_models_comparison.py` - Model comparison script
- `setup_gemini_env.sh` - Automated environment setup
- `main.py` - Original LLaVA-based pipeline (requires LLaVA server)

## Next Steps

1. Run the automated setup script
2. Set your Gemini API key
3. Test the Gemini inspection: `python main_gemini.py`
4. Optionally run model comparison: `python evaluate_models_comparison.py`

For questions or issues, refer to the conversation history or check the error messages carefully.
