# 🚀 Quick Start Guide

Get your Voice Summarizer Pro up and running in 5 minutes!

## Prerequisites Check

✅ Python 3.8+ installed  
✅ FFmpeg installed (for audio processing)  
✅ 4GB+ RAM available  
✅ Modern web browser  

## Installation (3 Steps)

### Step 1: Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Linux:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

### Step 2: Set Up Python Environment

```bash
# Navigate to project folder
cd voice-summarizer-pro

# Run the startup script
./start.sh         # macOS/Linux
start.bat          # Windows
```

That's it! The script handles everything:
- Creates virtual environment
- Installs all dependencies
- Downloads AI models
- Sets up directories
- Starts the server

### Step 3: Open the Application

1. **Wait for the server to start** (you'll see "Running on http://localhost:5000")
2. **Open `index.html`** in your web browser
3. **Start processing audio!**

## First Use

1. **Upload**: Drag & drop an audio file (or click to browse)
2. **Configure**: Choose summary type and options
3. **Process**: Click "Process Audio" button
4. **Wait**: Watch real-time progress (10-60 seconds)
5. **View**: See your results with summary, keywords, sentiment
6. **Export**: Download in your preferred format

## Supported Files

✅ MP3  
✅ WAV  
✅ OGG  
✅ FLAC  
✅ M4A  
✅ WebM  
✅ MP4 (audio only)  

## Default Settings

- **Model**: Base (good balance of speed/accuracy)
- **Summary**: Brief (2-3 sentences)
- **Features**: All enabled (keywords, sentiment, action items)
- **Language**: Auto-detect
- **Translation**: Off

## Troubleshooting

**"ModuleNotFoundError"**
→ Run `pip install -r requirements.txt`

**"FFmpeg not found"**
→ Install FFmpeg (see Step 1)

**"Port 5000 already in use"**
→ Stop other apps using port 5000 or change PORT in config.py

**"Out of memory"**
→ Use smaller model in config.py: `WHISPER_MODEL_SIZE = "tiny"`

**"Processing too slow"**
→ Enable GPU in config.py: `USE_GPU = True`

## Quick Tips

💡 **For faster processing**: Use "tiny" or "base" models  
💡 **For better accuracy**: Use "medium" or "large" models  
💡 **For meetings**: Enable action items extraction  
💡 **For videos**: Export as SRT subtitles  
💡 **For documents**: Export as DOCX or PDF  

## What's Next?

📖 Read `README.md` for detailed documentation  
🎯 Check `ENHANCEMENT_GUIDE.md` for feature overview  
⚙️ Customize `config.py` to your needs  
🚀 Start processing your audio files!  

## Need Help?

- Check the README.md for detailed info
- Review the ENHANCEMENT_GUIDE.md for features
- Look at config.py for customization options
- Check the troubleshooting section above

---

**You're all set! Enjoy your enhanced Voice Summarizer Pro! 🎙️✨**
