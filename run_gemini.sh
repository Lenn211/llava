#!/bin/bash
# One-command launcher for Gemini inspection

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}   Gemini Inspection Pipeline Launcher${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found.${NC}"
    echo -e "${YELLOW}Running setup script...${NC}"
    echo ""
    
    if [ ! -f "setup_gemini_env.sh" ]; then
        echo -e "${RED}Error: setup_gemini_env.sh not found!${NC}"
        exit 1
    fi
    
    chmod +x setup_gemini_env.sh
    ./setup_gemini_env.sh
    echo ""
fi

# Activate venv
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo -e "${RED}GEMINI_API_KEY not set!${NC}"
    echo ""
    echo "Please set your Gemini API key:"
    echo "  export GEMINI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or add to ~/.bashrc for permanent setup:"
    echo "  echo 'export GEMINI_API_KEY=\"your-key\"' >> ~/.bashrc"
    echo "  source ~/.bashrc"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ API key found${NC}"

# Check if main_gemini.py exists
if [ ! -f "main_gemini.py" ]; then
    echo -e "${RED}Error: main_gemini.py not found!${NC}"
    exit 1
fi

# Run the script
echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}Starting Gemini Inspection...${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

python main_gemini.py

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Inspection Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Results saved in: gemini_results/"
echo ""

deactivate
