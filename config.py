"""
Configuration management for Parakeet ASR Service
Loads settings from config.env file
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Load environment variables from config.env if it exists
CONFIG_ENV_PATH = BASE_DIR / "config.env"
if CONFIG_ENV_PATH.exists():
    with open(CONFIG_ENV_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                value = value.split('#')[0].strip()
                os.environ[key.strip()] = value

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8006"))

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# File Configuration
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "1024"))
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024  # Convert to bytes
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Chunking Configuration
CHUNK_THRESHOLD_MB = int(os.getenv("CHUNK_THRESHOLD_MB", "30"))
CHUNK_THRESHOLD = CHUNK_THRESHOLD_MB * 1024 * 1024  # Convert to bytes
CHUNK_DURATION = int(os.getenv("CHUNK_DURATION", "300"))  # 5 minutes in seconds
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "10"))  # 10 seconds overlap

# Segment generation settings (legacy, not used with TDT)
MAX_SEGMENT_DURATION = float(os.getenv("MAX_SEGMENT_DURATION", "10.0"))  # seconds
MIN_SEGMENT_DURATION = float(os.getenv("MIN_SEGMENT_DURATION", "0.5"))   # seconds (optional)
SEGMENT_PUNCTUATION = ['.', '?', '!']  # Can be extended

# Paths
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/parakeet_uploads")
LOG_DIR = os.getenv("LOG_DIR", str(BASE_DIR / "logs"))
LOG_FILE = os.path.join(LOG_DIR, "service.log")

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

# Create log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

