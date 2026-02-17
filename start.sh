#!/bin/bash

# Voice Summarizer Pro - Startup Script
# This script handles the complete startup process

echo "🎙️  Voice Summarizer Pro - Starting..."
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/lib/python*/site-packages/flask/__init__.py" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
fi

# Download NLTK data if needed
echo -e "${BLUE}Checking NLTK data...${NC}"
python3 -c "
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print('✓ NLTK data available')
except LookupError:
    print('Downloading NLTK data...')
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✓ NLTK data downloaded')
"

# Create necessary directories
echo -e "${BLUE}Setting up directories...${NC}"
mkdir -p uploads outputs temp cache
echo -e "${GREEN}✓ Directories ready${NC}"

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}⚠ FFmpeg not found!${NC}"
    echo -e "${YELLOW}Please install FFmpeg:${NC}"
    echo "  macOS: brew install ffmpeg"
    echo "  Ubuntu: sudo apt install ffmpeg"
    echo "  Windows: Download from https://ffmpeg.org"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ FFmpeg found${NC}"
fi

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Voice Summarizer Pro is starting!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${BLUE}Backend:${NC}  http://localhost:5000"
echo -e "${BLUE}Frontend:${NC} Open index.html in your browser"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Start the Flask server
python3 app.py
