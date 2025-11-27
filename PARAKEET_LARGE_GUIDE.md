# Parakeet Large ASR Model - Complete Deployment Guide

## Table of Contents
1. [Model Overview](#model-overview)
2. [Official Documentation](#official-documentation)
3. [How the Model Works](#how-the-model-works)
4. [Deployment Architecture Decision](#deployment-architecture-decision)
5. [Audio Input Methods](#audio-input-methods)
6. [Resource Requirements](#resource-requirements)
7. [Performance & Processing Time](#performance--processing-time)
8. [Implementation Guide](#implementation-guide)
9. [Local Usage](#local-usage)
10. [Production Considerations](#production-considerations)
11. [Troubleshooting](#troubleshooting)

---

## Model Overview

**Parakeet Large (CTC 1.1B)** is an advanced Automatic Speech Recognition (ASR) system developed collaboratively by **NVIDIA NeMo** and **Suno.ai**. 

### Key Specifications
- **Model Type**: FastConformer XXL with CTC (Connectionist Temporal Classification)
- **Parameters**: ~1.1 Billion
- **Primary Use Case**: English speech-to-text transcription
- **Output Format**: Lowercase English text
- **Architecture**: FastConformer encoder + CTC decoder
- **Training Data**: Extensive datasets including Granary and NeMo ASR Set 3.0

### Model Capabilities
- High-accuracy transcription of English speech
- Support for various audio qualities and speaking styles
- Robust handling of complex speech patterns
- Suitable for both batch and real-time transcription tasks

---

## Official Documentation

### Primary Resources
1. **Hugging Face Model Hub**: 
   - URL: https://huggingface.co/nvidia/parakeet-ctc-1.1b
   - Contains model weights, configuration, and usage examples

2. **NVIDIA NeMo Documentation**:
   - URL: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html
   - Comprehensive ASR toolkit documentation

3. **NeMo GitHub Repository**:
   - URL: https://github.com/NVIDIA/NeMo
   - Source code, examples, and community support

4. **Research Paper**:
   - ArXiv: https://arxiv.org/abs/2509.14128
   - Detailed architecture and training methodology

### Additional Resources
- **NVIDIA Developer Blog**: https://developer.nvidia.com/blog/pushing-the-boundaries-of-speech-recognition-with-nemo-parakeet-asr-models/
- **NeMo ASR Collection**: Access to various pre-trained ASR models

---

## How the Model Works

### Architecture Overview

```
Audio Input → Preprocessing → FastConformer Encoder → CTC Decoder → Text Output
```

### Detailed Processing Pipeline

1. **Audio Preprocessing**
   - Audio files are converted to mel-spectrograms
   - Normalization and feature extraction
   - Resampling to model's expected sample rate (typically 16kHz)

2. **FastConformer Encoder**
   - Processes audio features through multiple transformer layers
   - Captures contextual information from speech
   - Generates high-dimensional representations

3. **CTC Decoder**
   - Converts encoder outputs to text sequences
   - Handles variable-length inputs and outputs
   - Produces lowercase English text

4. **Post-processing**
   - Text formatting and cleanup
   - Optional punctuation restoration (if enabled)

### Model Characteristics
- **Input**: Audio files (WAV, MP3, FLAC, etc.)
- **Sample Rate**: 16kHz (preferred, but can handle resampling)
- **Output**: Plain text transcription in lowercase
- **Language**: Primarily English (trained on English datasets)

---

## Deployment Architecture Decision

### Option 1: vLLM (NOT RECOMMENDED for Parakeet)

#### Analysis
- **What is vLLM**: High-throughput inference engine for Large Language Models
- **Primary Use Case**: Text-based LLMs (GPT, LLaMA, Qwen, etc.)
- **ASR Support**: Limited to no native support for audio processing models
- **Compatibility**: Parakeet Large cannot be directly hosted via vLLM

#### Why vLLM Doesn't Work
1. vLLM is optimized for text token processing, not audio spectrograms
2. No built-in audio preprocessing pipeline
3. Architecture mismatch (transformer-decoder vs encoder-decoder ASR)
4. Missing audio feature extraction capabilities

### Option 2: FastAPI + NeMo (RECOMMENDED)

#### Why FastAPI is the Right Choice

✅ **Advantages:**
1. **Native Audio Support**: Can handle multipart file uploads easily
2. **Flexible Integration**: Works seamlessly with NeMo toolkit
3. **Custom Preprocessing**: Full control over audio processing pipeline
4. **Multiple Input Methods**: Supports file uploads and URL-based inputs
5. **Scalability**: Easy to containerize and deploy
6. **Async Support**: Non-blocking request handling for better throughput

✅ **NeMo Integration Benefits:**
- Official NVIDIA toolkit for ASR models
- Pre-built model loading and inference functions
- Optimized for GPU acceleration
- Comprehensive audio preprocessing utilities
- Easy model fine-tuning capabilities

#### Architecture Recommendation

```
Client → FastAPI Endpoint → Audio Validation → NeMo Model → Transcription Response
```

**Deployment Stack:**
- Web Framework: FastAPI
- ASGI Server: Uvicorn (with multiple workers)
- Model Framework: NVIDIA NeMo
- GPU Acceleration: PyTorch + CUDA
- Optional: Redis for caching, Celery for async processing

---

## Audio Input Methods

### Method 1: Direct File Upload (Recommended for API)

#### Implementation
```python
from fastapi import FastAPI, UploadFile, File
import shutil
import os

@app.post("/transcribe/upload")
async def transcribe_upload(file: UploadFile = File(...)):
    """
    Accept audio file upload via multipart/form-data
    """
    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Transcribe using NeMo
    transcription = model.transcribe([temp_path])
    
    # Cleanup
    os.remove(temp_path)
    
    return {"transcription": transcription[0]}
```

#### Advantages
- Simple client implementation
- No external hosting needed for audio files
- Better for sensitive/private audio data
- Immediate processing

#### Considerations
- File size limits (recommend max 100MB)
- Upload time for large files
- Temporary storage management
- Bandwidth usage

#### Client Usage Example
```bash
curl -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/audio.wav"
```

```python
import requests

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/transcribe/upload",
        files={"file": f}
    )
print(response.json())
```

### Method 2: URL-Based Input (Recommended for External Files)

#### Implementation
```python
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
import requests
import tempfile

class TranscribeRequest(BaseModel):
    audio_url: HttpUrl

@app.post("/transcribe/url")
async def transcribe_url(request: TranscribeRequest):
    """
    Accept URL to audio file and download for processing
    """
    # Download audio file
    response = requests.get(str(request.audio_url), timeout=30)
    response.raise_for_status()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(response.content)
        temp_path = temp_file.name
    
    # Transcribe
    transcription = model.transcribe([temp_path])
    
    # Cleanup
    os.remove(temp_path)
    
    return {"transcription": transcription[0]}
```

#### Advantages
- No file upload overhead for client
- Works well with CDN-hosted files
- Easier for distributed systems
- Can process files from cloud storage (S3, GCS)

#### Considerations
- Requires public/accessible URLs
- Additional download time
- Network reliability dependency
- Security: validate URL domains

#### Client Usage Example
```bash
curl -X POST "http://localhost:8000/transcribe/url" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.wav"}'
```

```python
import requests

response = requests.post(
    "http://localhost:8000/transcribe/url",
    json={"audio_url": "https://example.com/audio.wav"}
)
print(response.json())
```

### Method 3: Base64 Encoded Audio (For Small Files)

#### Implementation
```python
from pydantic import BaseModel
import base64
import tempfile

class TranscribeBase64Request(BaseModel):
    audio_base64: str
    format: str = "wav"

@app.post("/transcribe/base64")
async def transcribe_base64(request: TranscribeBase64Request):
    """
    Accept base64-encoded audio data
    """
    # Decode base64
    audio_data = base64.b64decode(request.audio_base64)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.format}") as temp_file:
        temp_file.write(audio_data)
        temp_path = temp_file.name
    
    # Transcribe
    transcription = model.transcribe([temp_path])
    
    # Cleanup
    os.remove(temp_path)
    
    return {"transcription": transcription[0]}
```

#### Advantages
- No separate file upload step
- Works in pure JSON APIs
- Easy for small audio clips

#### Limitations
- Not suitable for large files (>10MB)
- 33% size overhead from base64 encoding
- Slower parsing for large payloads

### Supported Audio Formats

The NeMo toolkit supports various audio formats through librosa/soundfile:

| Format | Extension | Recommended | Notes |
|--------|-----------|-------------|-------|
| WAV | .wav | ✅ Yes | Uncompressed, best quality |
| FLAC | .flac | ✅ Yes | Lossless compression |
| MP3 | .mp3 | ⚠️ Acceptable | Lossy compression |
| OGG | .ogg | ⚠️ Acceptable | Lossy compression |
| M4A | .m4a | ⚠️ Acceptable | Requires ffmpeg |

**Recommended Format**: WAV (16kHz, 16-bit, mono) for best results

---

## Resource Requirements

### GPU Requirements

#### Minimum Configuration (Development/Testing)
- **GPU**: 1x NVIDIA A100 (40GB) or equivalent
- **CPU**: 8 cores
- **RAM**: 32 GB
- **Storage**: 50 GB (model + temp files)
- **Throughput**: ~5-10 files concurrently

#### Recommended Configuration (Production)
- **GPU**: 1x NVIDIA H100 (80GB) 
- **CPU**: 16-32 cores
- **RAM**: 64-128 GB
- **Storage**: 100-200 GB SSD
- **Throughput**: 20-50 concurrent requests

#### High-Throughput Configuration
- **GPU**: 2-4x NVIDIA H100 (80GB each)
- **CPU**: 32-64 cores
- **RAM**: 128-256 GB
- **Storage**: 500 GB NVMe SSD
- **Load Balancer**: Nginx or HAProxy
- **Throughput**: 100+ concurrent requests

### Why H100 with 80GB?

#### Benefits for Parakeet Large (1.1B parameters)
1. **Model Size**: Full model in FP16 requires ~2.2GB, plenty of headroom
2. **Batch Processing**: Can process multiple audio files simultaneously
3. **Long Audio Files**: Sufficient memory for spectrograms of long recordings
4. **Future-Proofing**: Room for model updates or ensemble methods

#### GPU Utilization Estimates
- **Model Loading**: ~2-3 GB VRAM
- **Per-Request Overhead**: ~500MB - 2GB (depends on audio length)
- **Optimal Batch Size**: 8-16 files concurrently on H100 80GB

### CPU Core Allocation

#### Core Usage Breakdown
- **Audio Preprocessing**: 2-4 cores per request (resampling, feature extraction)
- **HTTP Handling**: 2-4 cores (FastAPI/Uvicorn workers)
- **Background Tasks**: 2-4 cores (cleanup, monitoring)
- **System Overhead**: 2-4 cores

**Recommendation**: For H100 80GB setup, use 16-32 CPU cores

### Memory (RAM) Requirements

#### Memory Usage Breakdown
- **Model Weights (CPU cache)**: ~2-4 GB
- **Audio File Buffers**: ~100-500 MB per concurrent request
- **System + FastAPI**: ~4-8 GB
- **Preprocessing Pipeline**: ~2-4 GB

**Formula**: RAM = 8GB (base) + (500MB × concurrent_requests)

For 50 concurrent requests: 8GB + 25GB = **33GB minimum**, recommend **64GB**

### Storage Requirements

#### Breakdown
- **Model Weights**: ~2-5 GB
- **NeMo Dependencies**: ~10-15 GB
- **PyTorch + CUDA**: ~5-10 GB
- **System Packages**: ~5-10 GB
- **Temporary Audio Files**: ~10-20 GB (with rotation)
- **Logs & Monitoring**: ~5-10 GB

**Total**: 50-80 GB minimum, **100-200 GB recommended**

### Network Requirements

- **Bandwidth**: 100 Mbps minimum, 1 Gbps recommended
- **Latency**: <50ms to clients (for real-time feel)
- **File Upload Limits**: Configure for 100-500 MB max file size

---

## Performance & Processing Time

### Processing Speed Factors

1. **Audio Length**: Primary factor - longer audio = more processing time
2. **GPU Model**: H100 >> A100 >> V100
3. **Batch Size**: Processing multiple files together improves throughput
4. **Audio Quality**: Higher sample rates require more processing
5. **System Load**: Concurrent requests affect individual processing times

### Time Estimates (NVIDIA H100 80GB)

#### Real-Time Factor (RTF)
RTF is the ratio of processing time to audio duration.
- **RTF = 0.1**: 1 minute audio processes in 6 seconds
- **RTF = 1.0**: 1 minute audio processes in 60 seconds

#### Expected Performance

| Audio Length | Processing Time | RTF | Notes |
|--------------|----------------|-----|-------|
| 30 seconds | 1-3 seconds | 0.03-0.10 | Optimal |
| 1 minute | 3-6 seconds | 0.05-0.10 | Fast |
| 5 minutes | 15-30 seconds | 0.05-0.10 | Good |
| 10 minutes | 30-60 seconds | 0.05-0.10 | Acceptable |
| 30 minutes | 1.5-3 minutes | 0.05-0.10 | Longer wait |
| 1 hour | 3-6 minutes | 0.05-0.10 | Batch recommended |

**Note**: These are estimates for H100 80GB. Actual performance may vary based on:
- Audio complexity (background noise, multiple speakers)
- System load (concurrent requests)
- Network I/O (file upload/download time)
- Preprocessing overhead

### Throughput Estimates

#### Single H100 80GB GPU

**Sequential Processing:**
- Short files (30s-1min): 600-1200 files/hour
- Medium files (5min): 150-200 files/hour
- Long files (30min): 20-40 files/hour

**Concurrent Processing (batch=8):**
- Short files: 2000-3000 files/hour
- Medium files: 400-600 files/hour
- Long files: 80-120 files/hour

#### Multi-GPU Setup (4x H100 80GB)

**Theoretical Maximum:**
- Short files: 8000-12000 files/hour
- Medium files: 1600-2400 files/hour
- Long files: 320-480 files/hour

### Optimization Strategies

1. **Batching**: Process multiple files together
2. **Async Processing**: Use Celery/RQ for background jobs
3. **Caching**: Cache results for identical audio files
4. **Preprocessing**: Optimize audio format conversion
5. **Model Optimization**: Use TensorRT or ONNX for faster inference

### Real-World Performance Example

**Scenario**: Podcast transcription service
- Average episode: 45 minutes
- Expected processing time: 2.5-4.5 minutes per episode
- With 1x H100 80GB: ~15-20 episodes/hour
- With 4x H100 80GB: ~60-80 episodes/hour

---

## Implementation Guide

### Prerequisites

```bash
# System requirements
- Ubuntu 20.04/22.04 or Rocky Linux 8/9
- NVIDIA Driver ≥ 525.x
- CUDA 12.1+
- Python 3.8-3.11
- Docker (optional, for containerized deployment)
```

### Step 1: Environment Setup

```bash
# Create project directory
mkdir -p /home/naresh/parakeet-service
cd /home/naresh/parakeet-service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install NVIDIA NeMo
pip install nemo_toolkit[asr]

# Install FastAPI and related packages
pip install fastapi uvicorn[standard] python-multipart pydantic
pip install aiofiles requests librosa soundfile

# Optional: For production
pip install gunicorn redis celery prometheus-client
```

Create `requirements.txt`:
```text
torch>=2.1.0
torchaudio>=2.1.0
nemo_toolkit[asr]>=1.22.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
python-multipart>=0.0.6
pydantic>=2.5.0
aiofiles>=23.2.1
requests>=2.31.0
librosa>=0.10.1
soundfile>=0.12.1
```

### Step 3: Create FastAPI Application

Create `app.py`:

```python
import os
import tempfile
import shutil
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
import requests
import nemo.collections.asr as nemo_asr

# Initialize FastAPI app
app = FastAPI(
    title="Parakeet Large ASR Service",
    description="Audio transcription service using NVIDIA Parakeet Large model",
    version="1.0.0"
)

# Global model variable
model = None

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
TEMP_DIR = "/tmp/parakeet_uploads"

# Create temp directory
os.makedirs(TEMP_DIR, exist_ok=True)

# Pydantic models
class TranscribeURLRequest(BaseModel):
    audio_url: HttpUrl
    
class TranscribeResponse(BaseModel):
    transcription: str
    audio_duration: Optional[float] = None
    processing_time: Optional[float] = None

# Model loading
@app.on_event("startup")
async def load_model():
    """Load the Parakeet Large model on startup"""
    global model
    print("Loading Parakeet Large model...")
    try:
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name="nvidia/parakeet-ctc-1.1b"
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

@app.on_event("shutdown")
async def cleanup():
    """Cleanup on shutdown"""
    # Clean up temp files
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

# Transcription endpoint - File Upload
@app.post("/transcribe/upload", response_model=TranscribeResponse)
async def transcribe_upload(file: UploadFile = File(...)):
    """
    Transcribe audio file uploaded directly
    
    Args:
        file: Audio file (WAV, MP3, FLAC, OGG, M4A)
        
    Returns:
        Transcription text
    """
    import time
    start_time = time.time()
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {ALLOWED_EXTENSIONS}"
        )
    
    # Create temp file
    temp_path = os.path.join(TEMP_DIR, f"{int(time.time())}_{file.filename}")
    
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Transcribe
        transcriptions = model.transcribe([temp_path])
        transcription_text = transcriptions[0]
        
        processing_time = time.time() - start_time
        
        return TranscribeResponse(
            transcription=transcription_text,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Transcription endpoint - URL
@app.post("/transcribe/url", response_model=TranscribeResponse)
async def transcribe_url(request: TranscribeURLRequest):
    """
    Transcribe audio file from URL
    
    Args:
        request: JSON with audio_url field
        
    Returns:
        Transcription text
    """
    import time
    start_time = time.time()
    
    temp_path = None
    
    try:
        # Download file from URL
        response = requests.get(str(request.audio_url), timeout=60, stream=True)
        response.raise_for_status()
        
        # Get file extension from URL or content-type
        url_path = Path(request.audio_url.path)
        file_ext = url_path.suffix.lower()
        
        if not file_ext or file_ext not in ALLOWED_EXTENSIONS:
            file_ext = ".wav"  # Default to wav
        
        # Create temp file
        temp_path = os.path.join(TEMP_DIR, f"{int(time.time())}_url_download{file_ext}")
        
        # Save downloaded content
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE/1024/1024}MB"
            )
        
        # Transcribe
        transcriptions = model.transcribe([temp_path])
        transcription_text = transcriptions[0]
        
        processing_time = time.time() - start_time
        
        return TranscribeResponse(
            transcription=transcription_text,
            processing_time=processing_time
        )
        
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# Batch transcription endpoint
@app.post("/transcribe/batch")
async def transcribe_batch(files: list[UploadFile] = File(...)):
    """
    Transcribe multiple audio files in batch
    
    Args:
        files: List of audio files
        
    Returns:
        List of transcriptions
    """
    import time
    start_time = time.time()
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per batch request"
        )
    
    temp_paths = []
    results = []
    
    try:
        # Save all uploaded files
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": f"Unsupported file format: {file_ext}"
                })
                continue
            
            temp_path = os.path.join(TEMP_DIR, f"{int(time.time())}_{file.filename}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_paths.append((file.filename, temp_path))
        
        # Transcribe all files in batch
        if temp_paths:
            paths_only = [path for _, path in temp_paths]
            transcriptions = model.transcribe(paths_only)
            
            for (filename, _), transcription in zip(temp_paths, transcriptions):
                results.append({
                    "filename": filename,
                    "status": "success",
                    "transcription": transcription
                })
        
        processing_time = time.time() - start_time
        
        return {
            "results": results,
            "total_files": len(files),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "processing_time": processing_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch transcription failed: {str(e)}")
    
    finally:
        # Cleanup all temp files
        for _, temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Step 4: Create Configuration File

Create `config.env`:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Model Configuration
MODEL_NAME=nvidia/parakeet-ctc-1.1b

# File Limits
MAX_FILE_SIZE_MB=100
MAX_BATCH_SIZE=10

# Paths
TEMP_DIR=/tmp/parakeet_uploads
LOG_DIR=/home/naresh/parakeet-service/logs

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### Step 5: Create Startup Script

Create `start_service.sh`:

```bash
#!/bin/bash

# Parakeet Large ASR Service Startup Script

# Load environment variables
if [ -f config.env ]; then
    export $(cat config.env | grep -v '^#' | xargs)
fi

# Activate virtual environment
source venv/bin/activate

# Create log directory
mkdir -p logs

# Start the service
echo "Starting Parakeet Large ASR Service..."
uvicorn app:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8000} \
    --workers ${WORKERS:-4} \
    --log-level info \
    --access-log \
    --log-config logging.conf \
    2>&1 | tee logs/service.log
```

Make it executable:
```bash
chmod +x start_service.sh
```

### Step 6: Create Logging Configuration

Create `logging.conf`:

```ini
[loggers]
keys=root,uvicorn,fastapi

[handlers]
keys=console,file

[formatters]
keys=default

[logger_root]
level=INFO
handlers=console,file

[logger_uvicorn]
level=INFO
handlers=console,file
qualname=uvicorn
propagate=0

[logger_fastapi]
level=INFO
handlers=console,file
qualname=fastapi
propagate=0

[handler_console]
class=StreamHandler
level=INFO
formatter=default
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=INFO
formatter=default
args=('logs/service.log', 'a')

[formatter_default]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

### Step 7: Create Test Client

Create `test_client.py`:

```python
import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

def test_upload(audio_file_path):
    """Test file upload transcription"""
    print(f"Testing file upload: {audio_file_path}")
    
    if not Path(audio_file_path).exists():
        print(f"Error: File not found: {audio_file_path}")
        return
    
    with open(audio_file_path, "rb") as f:
        files = {"file": (Path(audio_file_path).name, f)}
        response = requests.post(f"{BASE_URL}/transcribe/upload", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Transcription: {result['transcription']}")
        print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
    else:
        print(f"Error: {response.text}")
    print()

def test_url(audio_url):
    """Test URL-based transcription"""
    print(f"Testing URL: {audio_url}")
    
    payload = {"audio_url": audio_url}
    response = requests.post(f"{BASE_URL}/transcribe/url", json=payload)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Transcription: {result['transcription']}")
        print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
    else:
        print(f"Error: {response.text}")
    print()

if __name__ == "__main__":
    # Test health check
    test_health()
    
    # Test file upload if file path provided
    if len(sys.argv) > 1:
        test_upload(sys.argv[1])
    
    # Test URL if provided
    if len(sys.argv) > 2:
        test_url(sys.argv[2])
```

---

## Local Usage

### Starting the Service

```bash
cd /home/naresh/parakeet-service
source venv/bin/activate
./start_service.sh
```

The service will start on `http://localhost:8000`

### Testing from Local Machine

#### 1. Using cURL

**File Upload:**
```bash
curl -X POST "http://localhost:8000/transcribe/upload" \
  -F "file=@/path/to/your/audio.wav"
```

**URL-Based:**
```bash
curl -X POST "http://localhost:8000/transcribe/url" \
  -H "Content-Type: application/json" \
  -d '{"audio_url": "https://example.com/audio.wav"}'
```

**Batch Processing:**
```bash
curl -X POST "http://localhost:8000/transcribe/batch" \
  -F "files=@audio1.wav" \
  -F "files=@audio2.wav" \
  -F "files=@audio3.wav"
```

#### 2. Using Python Client

```python
import requests

# Single file upload
def transcribe_file(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:8000/transcribe/upload', files=files)
    return response.json()

# URL-based transcription
def transcribe_url(audio_url):
    response = requests.post(
        'http://localhost:8000/transcribe/url',
        json={'audio_url': audio_url}
    )
    return response.json()

# Batch transcription
def transcribe_batch(file_paths):
    files = [('files', open(fp, 'rb')) for fp in file_paths]
    response = requests.post('http://localhost:8000/transcribe/batch', files=files)
    # Close file handles
    for _, f in files:
        f.close()
    return response.json()

# Usage
result = transcribe_file('/home/naresh/datasets/audios/speech.wav')
print(result['transcription'])
```

#### 3. Using Test Script

```bash
# Test with local file
python test_client.py /home/naresh/datasets/videos/motivation.wav

# Test with URL
python test_client.py "" "https://example.com/sample.wav"
```

### Accessing from Remote Machine

If you want to access the service from another machine:

#### Option 1: Direct IP Access
```python
# Replace localhost with your server's IP
BASE_URL = "http://192.168.1.100:8000"
```

#### Option 2: Using Tailscale (Recommended)
Based on your existing Tailscale setup:

```python
# Use Tailscale hostname
BASE_URL = "http://your-tailscale-hostname:8000"
```

#### Option 3: SSH Tunnel
```bash
# On your local machine
ssh -L 8000:localhost:8000 user@remote-server

# Then access via localhost:8000
```

### API Documentation

Once the service is running, access interactive API docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Sample Audio Files for Testing

You can use your existing audio files:
```bash
/home/naresh/datasets/videos/motivation.wav
/home/naresh/datasets/videos/Bill-gates-thankyou.mp4  # Will extract audio
/home/naresh/datasets/videos/3-min-motivation.wav
```

---

## Production Considerations

### 1. Security Hardening

#### API Authentication
```python
from fastapi import Security, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API token"""
    token = credentials.credentials
    # Implement your token verification logic
    if token != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

# Protect endpoints
@app.post("/transcribe/upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    token: str = Security(verify_token)
):
    # ... transcription logic
```

#### Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/transcribe/upload")
@limiter.limit("10/minute")
async def transcribe_upload(request: Request, file: UploadFile = File(...)):
    # ... transcription logic
```

#### Input Validation
```python
import magic

def validate_audio_file(file_path: str) -> bool:
    """Validate file is actually an audio file"""
    mime = magic.from_file(file_path, mime=True)
    allowed_mimes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg']
    return mime in allowed_mimes
```

### 2. Scalability Solutions

#### Horizontal Scaling with Load Balancer

**Nginx Configuration** (`nginx.conf`):
```nginx
upstream parakeet_backend {
    least_conn;
    server 192.168.1.101:8000;
    server 192.168.1.102:8000;
    server 192.168.1.103:8000;
    server 192.168.1.104:8000;
}

server {
    listen 80;
    server_name transcribe.yourdomain.com;
    
    client_max_body_size 100M;
    
    location / {
        proxy_pass http://parakeet_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}
```

#### Async Processing with Celery

```python
from celery import Celery
import os

celery_app = Celery(
    'parakeet_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task
def transcribe_task(audio_path: str):
    """Background transcription task"""
    transcription = model.transcribe([audio_path])
    return transcription[0]

# In FastAPI endpoint
@app.post("/transcribe/async")
async def transcribe_async(file: UploadFile = File(...)):
    # Save file and queue task
    task = transcribe_task.delay(temp_path)
    return {"task_id": task.id, "status": "processing"}

@app.get("/transcribe/status/{task_id}")
async def get_task_status(task_id: str):
    task = transcribe_task.AsyncResult(task_id)
    return {"status": task.state, "result": task.result}
```

### 3. Monitoring & Observability

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
transcription_requests = Counter('transcription_requests_total', 'Total transcription requests')
transcription_duration = Histogram('transcription_duration_seconds', 'Transcription processing time')
transcription_errors = Counter('transcription_errors_total', 'Total transcription errors')

# Add metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# In your endpoints
@transcription_duration.time()
def process_transcription(audio_path):
    transcription_requests.inc()
    try:
        return model.transcribe([audio_path])
    except Exception as e:
        transcription_errors.inc()
        raise
```

#### Logging Best Practices
```python
import logging
import json

# Structured logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        return json.dumps(log_data)

# Apply formatter
handler = logging.FileHandler('logs/structured.log')
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
```

### 4. Caching Strategy

```python
import hashlib
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def get_audio_hash(audio_path: str) -> str:
    """Generate hash of audio file"""
    with open(audio_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

async def transcribe_with_cache(audio_path: str):
    """Transcribe with Redis caching"""
    audio_hash = get_audio_hash(audio_path)
    
    # Check cache
    cached = redis_client.get(f"transcription:{audio_hash}")
    if cached:
        return cached.decode('utf-8')
    
    # Transcribe
    transcription = model.transcribe([audio_path])[0]
    
    # Cache result (expire after 7 days)
    redis_client.setex(f"transcription:{audio_hash}", 604800, transcription)
    
    return transcription
```

### 5. Docker Deployment

**Dockerfile:**
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY config.env .

# Create temp directory
RUN mkdir -p /tmp/parakeet_uploads

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Docker Compose** (`docker-compose.yml`):
```yaml
version: '3.8'

services:
  parakeet:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./logs:/app/logs
      - /tmp/parakeet_uploads:/tmp/parakeet_uploads
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - parakeet
    restart: unless-stopped
```

### 6. Model Optimization

#### TensorRT Optimization (Advanced)
```python
# Export model to TensorRT for faster inference
# Note: This requires additional setup and testing

import torch_tensorrt

# After loading the model
optimized_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input(shape=[1, 80, 3000])],  # Example input shape
    enabled_precisions={torch.float16}
)
```

#### Mixed Precision Inference
```python
import torch

# Use automatic mixed precision
with torch.cuda.amp.autocast():
    transcription = model.transcribe([audio_path])
```

### 7. Backup and Disaster Recovery

```bash
# Regular model backup script
#!/bin/bash

BACKUP_DIR="/backup/parakeet"
MODEL_CACHE="~/.cache/huggingface"

# Backup model weights
rsync -av --progress $MODEL_CACHE $BACKUP_DIR/

# Backup configuration
cp /home/naresh/parakeet-service/config.env $BACKUP_DIR/

# Backup logs (last 30 days)
find /home/naresh/parakeet-service/logs -mtime -30 -type f -exec cp {} $BACKUP_DIR/logs/ \;
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. Model Loading Failures

**Problem**: `OSError: Can't load model nvidia/parakeet-ctc-1.1b`

**Solutions**:
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--nvidia--parakeet-ctc-1.1b

# Set Hugging Face token if needed
export HF_HOME=/home/naresh/.cache/huggingface
export HUGGINGFACE_TOKEN=your_token_here

# Manually download model
python -c "import nemo.collections.asr as nemo_asr; model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained('nvidia/parakeet-ctc-1.1b')"
```

#### 2. CUDA Out of Memory

**Problem**: `CUDA out of memory` errors

**Solutions**:
```python
# Reduce batch size
transcriptions = model.transcribe([audio_path], batch_size=1)

# Clear CUDA cache
import torch
torch.cuda.empty_cache()

# Process files sequentially instead of batch
for audio_file in audio_files:
    transcription = model.transcribe([audio_file])
    torch.cuda.empty_cache()
```

#### 3. Audio Format Issues

**Problem**: `Error loading audio file` or `Unsupported format`

**Solutions**:
```bash
# Install ffmpeg for format conversion
sudo apt-get install ffmpeg

# Convert audio to WAV
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Check audio file
ffprobe input.mp3
```

```python
# Add audio conversion in code
import subprocess

def convert_to_wav(input_path: str, output_path: str):
    """Convert audio to WAV format"""
    subprocess.run([
        'ffmpeg', '-i', input_path,
        '-ar', '16000',  # 16kHz sample rate
        '-ac', '1',       # Mono
        output_path
    ], check=True, capture_output=True)
```

#### 4. Slow Inference

**Problem**: Transcription takes too long

**Solutions**:
```python
# Use FP16 for faster inference
model = model.half()

# Ensure model is on GPU
model = model.cuda()

# Optimize for inference
model.eval()
with torch.no_grad():
    transcription = model.transcribe([audio_path])

# Check GPU utilization
nvidia-smi
```

#### 5. File Upload Timeout

**Problem**: Large files timeout during upload

**Solutions**:
```python
# Increase timeout in FastAPI
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# When running
uvicorn.run(app, timeout_keep_alive=300)  # 5 minutes

# In nginx
# proxy_read_timeout 300;
# client_body_timeout 300;
```

#### 6. Permission Errors

**Problem**: Cannot write to temp directory

**Solutions**:
```bash
# Fix permissions
sudo chmod 777 /tmp/parakeet_uploads

# Or use user-specific temp directory
mkdir -p ~/temp/parakeet_uploads
export TEMP_DIR=~/temp/parakeet_uploads
```

#### 7. Model Accuracy Issues

**Problem**: Poor transcription quality

**Solutions**:
```python
# Ensure audio quality
# - Sample rate: 16kHz
# - Channels: Mono
# - Format: WAV preferred

# Check audio preprocessing
import librosa
audio, sr = librosa.load(audio_path, sr=16000, mono=True)

# Try fine-tuning on domain-specific data
# See NeMo fine-tuning documentation
```

#### 8. Multi-GPU Setup Issues

**Problem**: Model not utilizing all GPUs

**Solutions**:
```python
# For multi-GPU inference
import torch

# Specify GPU
device = torch.device("cuda:0")
model = model.to(device)

# Or distribute across GPUs (requires data parallelism setup)
model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
```

### Debugging Tools

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Monitor GPU usage in real-time
watch -n 1 nvidia-smi

# Check NeMo installation
python -c "import nemo; print(nemo.__version__)"

# Test audio file loading
python -c "import librosa; audio, sr = librosa.load('test.wav'); print(f'Duration: {len(audio)/sr}s')"

# Check FastAPI logs
tail -f logs/service.log

# Test endpoint availability
curl http://localhost:8000/health
```

### Performance Profiling

```python
# Profile transcription function
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

transcription = model.transcribe([audio_path])

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

---

## Summary & Quick Reference

### Key Takeaways

1. **Model**: Parakeet Large (1.1B parameters) - NVIDIA NeMo ASR model
2. **Deployment**: FastAPI (vLLM not suitable for ASR models)
3. **Input Methods**: File upload, URL, Base64 (all supported)
4. **GPU**: 1x H100 80GB recommended, can use A100 40GB minimum
5. **CPU**: 16-32 cores for production
6. **RAM**: 64-128 GB
7. **Processing Time**: ~0.05-0.10 RTF (1 min audio = 3-6 seconds on H100)
8. **Throughput**: 150-1200 files/hour (depends on audio length)

### Quick Start Commands

```bash
# Setup
cd /home/naresh/parakeet-service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start service
./start_service.sh

# Test
python test_client.py /path/to/audio.wav

# API endpoints
http://localhost:8000/docs          # Interactive API docs
http://localhost:8000/health         # Health check
http://localhost:8000/transcribe/upload  # File upload
http://localhost:8000/transcribe/url     # URL-based
```

### Integration with Existing Services

Since you have multiple model services (FastAPI and vLLM-based), you can integrate Parakeet:

```python
# Example: Transcribe audio, then process with another model
import requests

# 1. Transcribe audio
audio_response = requests.post(
    'http://localhost:8000/transcribe/upload',
    files={'file': open('speech.wav', 'rb')}
)
transcription = audio_response.json()['transcription']

# 2. Process transcription with your Qwen model
qwen_response = requests.post(
    'http://localhost:8001/generate',  # Your Qwen service
    json={'text': transcription}
)

# 3. Use Omnivinci for multimodal analysis
omnivinci_response = requests.post(
    'http://localhost:8002/analyze',  # Your Omnivinci service
    json={'text': transcription, 'audio_url': 'original_audio.wav'}
)
```

### Cost Estimation (Cloud Deployment)

If deploying on cloud (e.g., AWS, GCP):

| Component | Specification | Estimated Cost (per hour) |
|-----------|---------------|---------------------------|
| GPU Instance | 1x H100 80GB | $20-30 |
| Storage | 200 GB SSD | $0.50 |
| Network | 1 TB transfer | $0.02/GB |
| Load Balancer | - | $0.50 |
| **Total** | - | **~$25-35/hour** |

**Monthly (730 hours)**: ~$18,000-25,000

**Cost per transcription** (5-min audio, 30s processing):
- Throughput: 120 files/hour
- Cost per file: ~$0.20-0.30

### Support & Resources

- **NeMo Documentation**: https://docs.nvidia.com/nemo-framework/
- **NeMo GitHub**: https://github.com/NVIDIA/NeMo
- **Hugging Face Model**: https://huggingface.co/nvidia/parakeet-ctc-1.1b
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **NVIDIA Developer Forums**: https://forums.developer.nvidia.com/

---

## Conclusion

This guide provides a complete overview of deploying the Parakeet Large model for audio transcription. The recommended architecture using FastAPI + NeMo provides:

✅ **Flexibility**: Multiple input methods (upload, URL, batch)  
✅ **Performance**: Fast inference with H100 GPUs  
✅ **Scalability**: Easy to scale horizontally  
✅ **Integration**: Simple REST API for any client  
✅ **Production-Ready**: With proper monitoring, security, and error handling  

For your specific use case with existing FastAPI and vLLM services, this Parakeet service integrates seamlessly and provides high-quality audio transcription capabilities.

**Next Steps**:
1. Set up the environment and install dependencies
2. Deploy the FastAPI application
3. Test with your audio files in `/home/naresh/datasets/`
4. Integrate with your existing model services
5. Monitor performance and optimize as needed

Good luck with your deployment! 🚀

