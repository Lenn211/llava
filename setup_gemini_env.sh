#!/bin/bash
# Automated setup script for Gemini-based inspection pipeline

set -e  # Exit on error

echo "========================================="
echo "Setting up Python environment for Gemini"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Install Python 3.9
echo -e "${YELLOW}Step 1: Installing Python 3.9...${NC}"
sudo apt update
sudo apt install -y python3.9 python3.9-venv python3.9-dev

# Step 2: Create new virtual environment with Python 3.9
echo -e "${YELLOW}Step 2: Creating new virtual environment...${NC}"
cd /home/rishan/Desktop/LLaVA

# Backup old venv if it exists
if [ -d "venv" ]; then
    echo "Backing up old virtual environment..."
    mv venv venv_backup_$(date +%Y%m%d_%H%M%S)
fi

# Create new venv with Python 3.9
python3.9 -m venv venv

# Step 3: Activate and upgrade pip
echo -e "${YELLOW}Step 3: Activating environment and upgrading pip...${NC}"
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Step 4: Install all required packages
echo -e "${YELLOW}Step 4: Installing required packages...${NC}"
echo "Installing core packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo "Installing YOLO and vision packages..."
pip install ultralytics opencv-python pillow

echo "Installing Gemini SDK..."
pip install google-generativeai

echo "Installing other dependencies..."
pip install requests python-dotenv

# Step 5: Verify installation
echo -e "${YELLOW}Step 5: Verifying installation...${NC}"
python --version
pip list | grep -E "(torch|ultralytics|google-generativeai|opencv)"

# Step 6: Check for API key
echo -e "${YELLOW}Step 6: Checking Gemini API key...${NC}"
if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${RED}WARNING: GEMINI_API_KEY environment variable not set!${NC}"
    echo -e "${YELLOW}Please set it by running:${NC}"
    echo "export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or add it to your ~/.bashrc:"
    echo "echo 'export GEMINI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
    echo "source ~/.bashrc"
else
    echo -e "${GREEN}GEMINI_API_KEY is set!${NC}"
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then you can run the Gemini inspection script:"
echo "  python main_gemini.py"
echo ""

deactivate 2>/dev/null || true
