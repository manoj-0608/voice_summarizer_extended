"""
Configuration file for Voice Summarizer Pro
Modify these settings to customize your installation
"""

# =============================================================================
# WHISPER MODEL CONFIGURATION
# =============================================================================

# Model size: "tiny", "base", "small", "medium", "large"
# Larger models = better accuracy but slower and more memory
WHISPER_MODEL_SIZE = "base"

# Supported languages for Whisper (None = auto-detect all)
# Limit this list if you only need specific languages
WHISPER_LANGUAGES = None  # or ["en", "es", "fr"] for specific languages

# =============================================================================
# NLP MODEL CONFIGURATION
# =============================================================================

# Summarization model
SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
# Alternatives: "t5-small", "t5-base", "google/pegasus-xsum"

# Sentiment analysis model
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
# Alternatives: "nlptown/bert-base-multilingual-uncased-sentiment"

# =============================================================================
# PROCESSING OPTIONS
# =============================================================================

# Maximum audio file size (in MB)
MAX_FILE_SIZE_MB = 100

# Maximum processing time per file (in seconds)
MAX_PROCESSING_TIME = 600  # 10 minutes

# Enable GPU acceleration if available
USE_GPU = True

# Number of worker threads for batch processing
WORKER_THREADS = 2

# =============================================================================
# SUMMARY CONFIGURATION
# =============================================================================

# Summary length settings
SUMMARY_LENGTHS = {
    'brief': {
        'max_length': 100,
        'min_length': 30
    },
    'detailed': {
        'max_length': 300,
        'min_length': 100
    },
    'bullet_points': {
        'max_length': 150,
        'min_length': 50
    }
}

# Number of keywords to extract
NUM_KEYWORDS = 10

# Maximum number of action items to extract
MAX_ACTION_ITEMS = 10

# =============================================================================
# LANGUAGE SUPPORT
# =============================================================================

# Translation service
# Options: "google", "deepl" (requires API key)
TRANSLATION_SERVICE = "google"

# DeepL API key (if using DeepL)
DEEPL_API_KEY = None

# Supported translation languages
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish'
}

# =============================================================================
# FILE STORAGE
# =============================================================================

# Directory for uploaded files
UPLOAD_FOLDER = 'uploads'

# Directory for output files
OUTPUT_FOLDER = 'outputs'

# Directory for temporary files
TEMP_FOLDER = 'temp'

# History file location
HISTORY_FILE = 'history.json'

# Maximum number of history entries to keep
MAX_HISTORY_ENTRIES = 100

# Auto-delete old files after N days (0 = never delete)
AUTO_DELETE_DAYS = 30

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

# Enable/disable specific export formats
EXPORT_FORMATS = {
    'txt': True,
    'docx': True,
    'pdf': True,
    'srt': True,
    'json': True,
    'vtt': False,  # WebVTT subtitles
    'html': False  # HTML report
}

# PDF settings
PDF_PAGE_SIZE = 'letter'  # or 'a4'
PDF_FONT_SIZE = 11

# DOCX settings
DOCX_FONT = 'Calibri'
DOCX_FONT_SIZE = 11

# =============================================================================
# AUDIO PROCESSING
# =============================================================================

# Sample rate for audio processing (Hz)
AUDIO_SAMPLE_RATE = 16000

# Audio normalization
NORMALIZE_AUDIO = True

# Noise reduction
ENABLE_NOISE_REDUCTION = False

# Voice activity detection
ENABLE_VAD = False

# =============================================================================
# PERFORMANCE TUNING
# =============================================================================

# Chunk size for processing long audio (in seconds)
# Larger chunks = faster but more memory
AUDIO_CHUNK_SIZE = 30

# Enable parallel processing for chunks
PARALLEL_CHUNKS = False

# Cache transcriptions (saves re-processing same files)
ENABLE_CACHE = True
CACHE_FOLDER = 'cache'

# =============================================================================
# API CONFIGURATION
# =============================================================================

# Flask server settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

# Enable CORS for frontend
CORS_ENABLED = True

# WebSocket settings
SOCKETIO_ASYNC_MODE = 'threading'  # or 'eventlet', 'gevent'

# Rate limiting (requests per minute)
RATE_LIMIT = 60

# =============================================================================
# SECURITY
# =============================================================================

# Allowed file extensions
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a', 'webm', 'mp4'}

# Enable file upload validation
VALIDATE_UPLOADS = True

# Maximum concurrent users
MAX_CONCURRENT_USERS = 10

# Session timeout (minutes)
SESSION_TIMEOUT = 60

# =============================================================================
# LOGGING
# =============================================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
LOG_LEVEL = "INFO"

# Log file location
LOG_FILE = "voice_summarizer.log"

# Enable detailed logging
VERBOSE_LOGGING = False

# =============================================================================
# EXPERIMENTAL FEATURES
# =============================================================================

# Enable speaker diarization (who spoke when)
ENABLE_DIARIZATION = False

# Number of speakers (for diarization)
NUM_SPEAKERS = 2

# Enable real-time processing
ENABLE_REALTIME = False

# Enable audio enhancement
ENABLE_AUDIO_ENHANCEMENT = False

# =============================================================================
# UI CUSTOMIZATION
# =============================================================================

# Application name
APP_NAME = "Voice Summarizer Pro"

# Theme colors (used in frontend)
THEME = {
    'primary': '#FF6B35',
    'secondary': '#004E89',
    'accent': '#F7B801',
    'success': '#06D6A0',
    'danger': '#EF476F'
}

# Enable dark mode
DARK_MODE_AVAILABLE = True

# =============================================================================
# INTEGRATIONS
# =============================================================================

# Google Drive integration
GOOGLE_DRIVE_ENABLED = False
GOOGLE_DRIVE_CREDENTIALS = None

# Dropbox integration
DROPBOX_ENABLED = False
DROPBOX_ACCESS_TOKEN = None

# Slack notifications
SLACK_ENABLED = False
SLACK_WEBHOOK_URL = None

# Email notifications
EMAIL_ENABLED = False
EMAIL_CONFIG = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': None,
    'password': None,
    'from_address': None
}

# =============================================================================
# CUSTOM PROMPTS
# =============================================================================

# Custom system prompts for different summary types
CUSTOM_PROMPTS = {
    'brief': "Provide a concise summary in 2-3 sentences highlighting the main points.",
    'detailed': "Provide a comprehensive summary covering all important details and context.",
    'bullet_points': "Extract the key points as a bullet-point list.",
    'action_items': "Identify and list all action items and tasks mentioned.",
    'meeting_notes': "Format this as meeting notes with attendees, topics, decisions, and action items."
}

# =============================================================================
# ADVANCED SETTINGS
# =============================================================================

# Confidence threshold for language detection
LANGUAGE_CONFIDENCE_THRESHOLD = 0.7

# Minimum audio duration (seconds)
MIN_AUDIO_DURATION = 1

# Maximum audio duration (seconds, 0 = no limit)
MAX_AUDIO_DURATION = 3600  # 1 hour

# Silence threshold for VAD (dB)
SILENCE_THRESHOLD = -40

# Enable model quantization for faster inference
ENABLE_QUANTIZATION = False

# Mixed precision training (for fine-tuning)
USE_MIXED_PRECISION = False
