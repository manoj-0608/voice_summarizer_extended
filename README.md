# 🎙️ Voice Summarizer Pro

An advanced voice transcription and summarization system with multi-language support, sentiment analysis, and beautiful visualizations.

## ✨ Features

### Core Features
- **🎯 Whisper-powered Transcription** - High-accuracy audio transcription
- **📝 Multi-type Summaries** - Brief, detailed, or bullet-point summaries
- **🌍 Multi-language Support** - Auto-detection and translation for 20+ languages
- **🔑 Keyword Extraction** - Automatic identification of key terms
- **😊 Sentiment Analysis** - Understand the emotional tone
- **✅ Action Items** - Extract actionable tasks from audio
- **🎨 Beautiful Modern UI** - Gradient-rich, animated interface
- **📊 Audio Visualization** - Waveform display and analysis
- **💾 Multiple Export Formats** - TXT, DOCX, PDF, SRT, JSON

### Advanced Features
- **Real-time Progress Updates** - WebSocket-powered live progress
- **Processing History** - Keep track of all processed files
- **Batch Processing Ready** - Architecture supports multiple files
- **Speaker Diarization Ready** - Extendable for multi-speaker scenarios
- **Customizable Options** - Control what features to enable
- **Professional Exports** - Publication-ready documents

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- FFmpeg (for audio processing)
- 4GB+ RAM recommended
- Modern web browser

### Installation

1. **Clone or download the project**
```bash
cd voice-summarizer-pro
```

2. **Install FFmpeg** (if not already installed)

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from https://ffmpeg.org/download.html and add to PATH

3. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

5. **Download NLTK data** (automatic on first run)
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Running the Application

1. **Start the Flask backend**
```bash
python app.py
```
The server will start on `http://localhost:5000`

2. **Open the frontend**
- Open `index.html` in your web browser
- Or serve it with a simple HTTP server:
```bash
python -m http.server 8000
```
Then visit `http://localhost:8000`

## 📖 Usage Guide

### Basic Workflow

1. **Upload Audio**
   - Drag and drop an audio file onto the upload zone
   - Or click to browse and select a file
   - Supported formats: MP3, WAV, OGG, FLAC, M4A, WebM

2. **Configure Options**
   - Choose summary type (Brief, Detailed, or Bullet Points)
   - Select target language for translation (optional)
   - Enable/disable features like keywords, sentiment analysis

3. **Process**
   - Click "Process Audio"
   - Watch real-time progress as it:
     - Analyzes audio features
     - Transcribes speech
     - Detects language
     - Generates summary
     - Extracts insights

4. **View Results**
   - Summary and full transcript
   - Keywords and action items
   - Sentiment analysis
   - Translation (if requested)
   - Audio metadata

5. **Export**
   - Download in your preferred format
   - Options: TXT, DOCX, PDF, SRT subtitles, JSON

### Advanced Features

#### Multi-language Support
The system automatically detects the language of your audio and can translate summaries to 20+ languages including:
- Spanish, French, German, Italian, Portuguese
- Russian, Polish, Dutch, Swedish, Danish
- Japanese, Korean, Chinese (Simplified & Traditional)
- Arabic, Hindi, Turkish, Finnish, Norwegian

#### Sentiment Analysis
Get insights into the emotional tone of your audio:
- Positive/Negative classification
- Confidence scores
- Useful for customer service, feedback analysis, meeting notes

#### Action Items Extraction
Automatically identifies actionable tasks from phrases like:
- "need to", "should", "must", "have to"
- "will", "going to", "plan to"
- "action", "follow up"

#### Export Formats

**TXT** - Simple plain text with metadata  
**DOCX** - Formatted Word document with sections and styling  
**PDF** - Professional PDF with tables and formatting  
**SRT** - Subtitle file with timestamps for video  
**JSON** - Complete data export for programmatic use  

## 🏗️ Architecture

### Backend (Flask)
```
app.py
├── Whisper Model Loading (lazy)
├── Audio Processing
│   ├── Feature Extraction (librosa)
│   ├── Waveform Analysis
│   └── Format Conversion (pydub)
├── NLP Pipeline
│   ├── Transcription (Whisper)
│   ├── Summarization (BART)
│   ├── Sentiment Analysis (Transformers)
│   └── Keyword Extraction (NLTK)
├── Translation (Google Translate)
├── Export Generation
│   ├── DOCX (python-docx)
│   ├── PDF (ReportLab)
│   └── SRT (custom)
└── WebSocket (Socket.IO)
```

### Frontend (React)
```
index.html
├── File Upload (drag & drop)
├── Options Configuration
├── Real-time Progress
├── Results Visualization
│   ├── Metadata Cards
│   ├── Summary Display
│   ├── Keyword Tags
│   ├── Action Items List
│   └── Transcript Viewer
├── Export Buttons
└── Processing History
```

## 🎨 Design Philosophy

The UI follows these principles:
- **Bold, Distinctive Aesthetics** - Gradient-rich design that stands out
- **Smooth Animations** - Every interaction feels delightful
- **Intuitive Flow** - Clear progression from upload to results
- **Responsive Layout** - Works on desktop and mobile
- **Professional Output** - Export-ready documents

## 🔧 Customization

### Change Whisper Model Size
In `app.py`, line 84:
```python
whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
```

Larger models = better accuracy but slower processing

### Add Custom Languages
In `app.py`, `get_supported_languages()` function

### Modify Summary Length
In `app.py`, `generate_summary()` function, adjust `max_length` and `min_length`

### Customize UI Colors
In `index.html`, modify CSS variables in `:root`:
```css
--primary: #FF6B35;
--secondary: #004E89;
--accent: #F7B801;
```

## 📊 Performance Tips

1. **For faster processing:**
   - Use smaller Whisper model (tiny or base)
   - Disable features you don't need
   - Convert audio to 16kHz mono before uploading

2. **For better accuracy:**
   - Use larger Whisper model (medium or large)
   - Ensure clear audio quality
   - Remove background noise before uploading

3. **Memory optimization:**
   - Close unused tabs when processing large files
   - Clear history periodically
   - Process files one at a time

## 🛠️ Troubleshooting

### Common Issues

**"No module named 'whisper'"**
```bash
pip install openai-whisper
```

**"FFmpeg not found"**
Install FFmpeg using the instructions in Prerequisites

**WebSocket connection failed**
- Ensure Flask server is running
- Check if port 5000 is available
- Try `http://localhost:5000` instead of `127.0.0.1`

**Out of memory errors**
- Use smaller Whisper model
- Process shorter audio files
- Increase system RAM or use cloud deployment

**Translation not working**
- Check internet connection (Google Translate API requires network)
- Try alternative translation library if needed

## 🚀 Future Enhancements

Potential features to add:
- [ ] Speaker diarization (identify different speakers)
- [ ] Real-time audio recording and processing
- [ ] Batch processing multiple files
- [ ] Custom vocabulary and domain adaptation
- [ ] Integration with cloud storage (Dropbox, Google Drive)
- [ ] API endpoint for programmatic access
- [ ] Mobile app versions
- [ ] Meeting minutes templates
- [ ] Automatic summarization scheduling
- [ ] Audio enhancement and noise reduction

## 📝 API Documentation

### POST `/api/transcribe`
Upload and process audio file

**Request:**
- `file`: Audio file (multipart/form-data)
- `summary_type`: "brief" | "detailed" | "bullet_points"
- `include_keywords`: boolean
- `include_sentiment`: boolean
- `include_action_items`: boolean
- `translate_to`: language code (optional)

**Response:**
```json
{
  "id": "uuid",
  "filename": "audio.mp3",
  "transcript": "...",
  "summary": "...",
  "language": "en",
  "duration": 120.5,
  "keywords": [...],
  "action_items": [...],
  "sentiment": {...}
}
```

### POST `/api/export/{file_id}`
Export results in various formats

**Request:**
```json
{
  "format": "txt|docx|pdf|srt|json",
  "data": { /* result object */ }
}
```

**Response:** File download

### GET `/api/history`
Get processing history

**Response:**
```json
[
  {
    "id": "uuid",
    "filename": "audio.mp3",
    "timestamp": "2024-01-01T12:00:00",
    "language": "en",
    "duration": 120.5
  }
]
```

### POST `/api/translate`
Translate text

**Request:**
```json
{
  "text": "Hello world",
  "target_language": "es"
}
```

**Response:**
```json
{
  "text": "Hola mundo",
  "source_language": "en",
  "target_language": "es",
  "confidence": 0.9
}
```

## 🤝 Contributing

Contributions are welcome! Areas where you can help:
- Add support for more audio formats
- Improve summarization algorithms
- Create additional export templates
- Add more language support
- Optimize performance
- Write tests
- Improve documentation

## 📄 License

This project is provided as-is for educational and commercial use.

## 🙏 Acknowledgments

- OpenAI Whisper for speech recognition
- Hugging Face Transformers for NLP models
- Flask and React communities
- All open-source contributors

## 📞 Support

For issues, questions, or suggestions:
- Check the Troubleshooting section
- Review closed issues on GitHub
- Open a new issue with details

---

**Made with ❤️ for better audio understanding**
