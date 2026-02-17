@echo off
echo ============================================
echo   Voice Summarizer - Smart Install Script
echo   Works with Python 3.13
echo ============================================
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Upgrade pip and build tools first
echo [1/10] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel
echo Done!
echo.

REM Install Flask and web packages (always works)
echo [2/10] Installing Flask and web packages...
pip install Flask flask-cors flask-socketio python-socketio
echo Done!
echo.

REM Install PyTorch CPU version (smaller, more compatible)
echo [3/10] Installing PyTorch (CPU version)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
echo Done!
echo.

REM Install Whisper
echo [4/10] Installing Whisper...
pip install openai-whisper
echo Done!
echo.

REM Install transformers WITHOUT sentencepiece first
echo [5/10] Installing Transformers...
pip install transformers
echo Done!
echo.

REM Try sentencepiece (may fail on Python 3.13 - that's OK)
echo [6/10] Installing sentencepiece (optional)...
pip install sentencepiece
if errorlevel 1 (
    echo sentencepiece failed - skipping, not critical
) else (
    echo Done!
)
echo.

REM Install audio libraries (no pydub!)
echo [7/10] Installing audio libraries...
pip install librosa soundfile
echo Done!
echo.

REM Install language tools
echo [8/10] Installing language tools...
pip install langdetect
pip install googletrans==4.0.0rc1
pip install nltk
echo Done!
echo.

REM Install document generation
echo [9/10] Installing document generation...
pip install python-docx reportlab
echo Done!
echo.

REM Install data science tools
echo [10/10] Installing data science tools...
pip install numpy scipy pandas
echo Done!
echo.

REM Download NLTK data
echo Downloading NLTK data...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab'); print('NLTK data downloaded!')"
echo.

REM Test imports
echo ============================================
echo   Testing Imports...
echo ============================================
python -c "import flask; print('OK Flask')"
python -c "import flask_socketio; print('OK Flask-SocketIO')"
python -c "import whisper; print('OK Whisper')"
python -c "import transformers; print('OK Transformers')"
python -c "import librosa; print('OK Librosa')"
python -c "import nltk; print('OK NLTK')"
python -c "import langdetect; print('OK Langdetect')"
python -c "import docx; print('OK python-docx')"
python -c "import reportlab; print('OK ReportLab')"
echo.

echo ============================================
echo   Installation Complete!
echo   Now run: python app.py
echo ============================================
pause