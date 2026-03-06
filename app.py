"""
Parakeet TDT ASR Service
FastAPI application for audio transcription using NVIDIA Parakeet TDT model
"""
import os
import time
import logging
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import unquote, quote

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import requests
import librosa
import soundfile as sf
import nemo.collections.asr as nemo_asr

import config
from segment_utils import calculate_metadata, merge_segment_boundaries

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Parakeet TDT ASR Service",
    description="Audio transcription service using NVIDIA Parakeet TDT model",
    version="3.0.0"
)

# Global model variable
model = None


def preprocess_audio(audio_path: str) -> str:
    """
    Preprocess audio file to ensure it's in the correct format for the model.
    Converts stereo to mono and resamples to 16kHz if needed.
    
    Args:
        audio_path: Path to the input audio file
        
    Returns:
        Path to the preprocessed audio file (temporary file)
    """
    try:
        # Load audio file - librosa automatically converts to mono and handles resampling
        target_sr = 16000
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        
        # Ensure audio is 1D array (mono)
        if len(audio.shape) > 1:
            logger.warning(f"Audio has unexpected shape: {audio.shape}, flattening...")
            audio = audio.flatten()
        
        # Create temporary file for preprocessed audio
        temp_fd, temp_path = tempfile.mkstemp(suffix='.wav', dir=config.TEMP_DIR)
        os.close(temp_fd)
        
        # Save preprocessed audio as WAV file (mono, 16kHz)
        sf.write(temp_path, audio, target_sr, format='WAV', subtype='PCM_16')
        logger.info(f"Preprocessed audio saved to: {temp_path} (shape: {audio.shape}, sr: {target_sr}Hz)")
        
        return temp_path
        
    except Exception as e:
        logger.error(f"Audio preprocessing failed: {str(e)}")
        raise


def split_audio_into_chunks(audio_path: str, chunk_duration: int = None, overlap: int = None) -> List[Dict[str, Any]]:
    """
    Split large audio file into smaller chunks with overlap for processing
    
    Args:
        audio_path: Path to preprocessed audio file
        chunk_duration: Duration of each chunk in seconds (default from config)
        overlap: Overlap duration in seconds (default from config)
        
    Returns:
        List of dicts with 'path' (chunk file path) and 'offset' (time offset in seconds)
    """
    if chunk_duration is None:
        chunk_duration = config.CHUNK_DURATION
    if overlap is None:
        overlap = config.CHUNK_OVERLAP
    
    try:
        # Get audio duration without loading full file
        audio_info = sf.info(audio_path)
        total_duration = audio_info.duration
        sample_rate = audio_info.samplerate
        
        logger.info(f"Splitting audio: {total_duration:.2f}s duration, {chunk_duration}s chunks, {overlap}s overlap")
        
        # Calculate number of chunks needed
        effective_chunk_duration = chunk_duration - overlap  # Overlap is subtracted from subsequent chunks
        num_chunks = int((total_duration - overlap) / effective_chunk_duration) + 1
        
        if num_chunks <= 1:
            logger.info("Audio shorter than chunk duration, no splitting needed")
            return [{'path': audio_path, 'offset': 0.0}]
        
        chunks = []
        
        # Read audio data once
        audio_data, sr = sf.read(audio_path)
        samples_per_second = sr
        
        for i in range(num_chunks):
            # Calculate start and end times for this chunk
            if i == 0:
                # First chunk: no overlap at start
                start_time = 0
                end_time = chunk_duration
            elif i == num_chunks - 1:
                # Last chunk: extends to end of file
                start_time = i * effective_chunk_duration
                end_time = total_duration
            else:
                # Middle chunks: overlap on both sides
                start_time = i * effective_chunk_duration
                end_time = start_time + chunk_duration
            
            # Convert to sample indices
            start_sample = int(start_time * samples_per_second)
            end_sample = int(end_time * samples_per_second)
            end_sample = min(end_sample, len(audio_data))  # Don't exceed audio length
            
            # Extract chunk
            chunk_audio = audio_data[start_sample:end_sample]
            
            # Save chunk to temp file
            chunk_filename = f"chunk_{int(time.time())}_{i:03d}.wav"
            chunk_path = os.path.join(config.TEMP_DIR, chunk_filename)
            sf.write(chunk_path, chunk_audio, sr, format='WAV', subtype='PCM_16')
            
            # Calculate time offset for this chunk (for timestamp adjustment)
            time_offset = i * effective_chunk_duration
            
            chunks.append({
                'path': chunk_path,
                'offset': time_offset,
                'chunk_index': i,
                'start_time': start_time,
                'end_time': end_time
            })
            
            logger.info(f"Created chunk {i+1}/{num_chunks}: {start_time:.2f}s-{end_time:.2f}s (offset: {time_offset:.2f}s)")
        
        logger.info(f"Split audio into {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Audio splitting failed: {str(e)}")
        raise


def process_single_chunk(chunk_path: str, time_offset: float) -> Dict[str, Any]:
    """
    Process a single audio chunk and adjust timestamps with offset
    
    Args:
        chunk_path: Path to chunk audio file
        time_offset: Time offset to add to all timestamps (in seconds)
        
    Returns:
        Dict with transcription, word_timestamps, segment_timestamps
    """
    try:
        # Transcribe chunk with timestamps
        transcriptions = model.transcribe([chunk_path], timestamps=True)
        result = transcriptions[0]
        
        # Extract text
        transcription_text = result.text if hasattr(result, 'text') else str(result)
        
        # Extract timestamps
        word_timestamps = extract_word_timestamps(result)
        segment_timestamps = extract_segment_timestamps(result)
        
        # Apply time offset to all timestamps
        for word in word_timestamps:
            word['start'] += time_offset
            word['end'] += time_offset
        
        for segment in segment_timestamps:
            segment['start'] += time_offset
            segment['end'] += time_offset
        
        logger.info(f"Chunk processed: {len(word_timestamps)} words, {len(segment_timestamps)} segments (offset: {time_offset:.2f}s)")
        
        return {
            'transcription': transcription_text,
            'word_timestamps': word_timestamps,
            'segment_timestamps': segment_timestamps
        }
        
    except Exception as e:
        logger.error(f"Chunk processing failed: {str(e)}")
        raise


def deduplicate_overlap(prev_chunk_words: List[Dict[str, Any]], curr_chunk_words: List[Dict[str, Any]], overlap_duration: float = None) -> List[Dict[str, Any]]:
    """
    Remove duplicate words from overlap region between chunks
    
    Args:
        prev_chunk_words: Word timestamps from previous chunk
        curr_chunk_words: Word timestamps from current chunk
        overlap_duration: Duration of overlap in seconds (default from config)
        
    Returns:
        Deduplicated word list for current chunk (duplicates removed)
    """
    if overlap_duration is None:
        overlap_duration = config.CHUNK_OVERLAP
    
    if not prev_chunk_words or not curr_chunk_words:
        return curr_chunk_words
    
    # Get the last words from previous chunk (in overlap region)
    prev_overlap_start = prev_chunk_words[-1]['end'] - overlap_duration
    prev_overlap_words = [w for w in prev_chunk_words if w['end'] >= prev_overlap_start]
    
    # Get first words from current chunk (in overlap region)
    curr_overlap_end = curr_chunk_words[0]['start'] + overlap_duration
    curr_overlap_words_indices = [i for i, w in enumerate(curr_chunk_words) if w['start'] <= curr_overlap_end]
    
    if not prev_overlap_words or not curr_overlap_words_indices:
        return curr_chunk_words
    
    # Find matching words to remove
    words_to_remove = set()
    
    for curr_idx in curr_overlap_words_indices:
        curr_word = curr_chunk_words[curr_idx]
        
        # Check if this word matches any in previous chunk's overlap
        for prev_word in prev_overlap_words:
            # Match criteria: similar text and close timestamps
            text_match = curr_word['word'].lower() == prev_word['word'].lower()
            # Check if timestamps are within reasonable proximity (accounting for offset differences)
            # Since offsets are already applied, we compare the actual timestamp values
            time_diff = abs(curr_word['start'] - prev_word['start'])
            time_match = time_diff < 0.5  # Within 0.5 seconds
            
            if text_match and time_match:
                words_to_remove.add(curr_idx)
                break
    
    # Remove duplicates
    deduplicated = [w for i, w in enumerate(curr_chunk_words) if i not in words_to_remove]
    
    if words_to_remove:
        logger.info(f"Removed {len(words_to_remove)} duplicate words from overlap region")
    
    return deduplicated


def merge_transcription_chunks(chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge results from multiple chunks into single transcription result
    
    Args:
        chunk_results: List of chunk processing results, each with transcription, word_timestamps, segment_timestamps
        
    Returns:
        Merged result with combined transcription, deduplicated words, and merged segments
    """
    if not chunk_results:
        raise ValueError("No chunk results to merge")
    
    if len(chunk_results) == 1:
        # Only one chunk, return as-is
        return chunk_results[0]
    
    logger.info(f"Merging {len(chunk_results)} chunks into single result")
    
    # Initialize with first chunk
    merged_transcription_parts = [chunk_results[0]['transcription']]
    merged_words = chunk_results[0]['word_timestamps'].copy()
    merged_segments = chunk_results[0]['segment_timestamps'].copy()
    
    # Process remaining chunks
    for i in range(1, len(chunk_results)):
        curr_chunk = chunk_results[i]
        
        # Deduplicate words in overlap region
        deduplicated_words = deduplicate_overlap(
            merged_words,
            curr_chunk['word_timestamps'],
            overlap_duration=config.CHUNK_OVERLAP
        )
        
        # Merge segment boundaries if needed
        merged_curr_segments = merge_segment_boundaries(
            merged_segments,
            curr_chunk['segment_timestamps']
        )
        
        # Append deduplicated words
        merged_words.extend(deduplicated_words)
        
        # Append segments (potentially with first segment merged into previous)
        merged_segments.extend(merged_curr_segments)
        
        # Append transcription text
        merged_transcription_parts.append(curr_chunk['transcription'])
    
    # Combine transcription text
    merged_transcription = ' '.join(merged_transcription_parts)
    
    logger.info(f"Merged result: {len(merged_words)} total words, {len(merged_segments)} total segments")
    
    return {
        'transcription': merged_transcription,
        'word_timestamps': merged_words,
        'segment_timestamps': merged_segments
    }


def extract_word_timestamps(hypothesis) -> List[Dict[str, Any]]:
    """
    Extract word-level timestamps from TDT model output
    
    Args:
        hypothesis: Hypothesis object from model.transcribe()
    
    Returns:
        List of word timestamp dictionaries
    """
    word_timestamps = []
    
    try:
        # TDT model provides timestamps directly via timestamp attribute
        if hasattr(hypothesis, 'timestamp') and 'word' in hypothesis.timestamp:
            word_data = hypothesis.timestamp['word']
            if isinstance(word_data, list):
                for item in word_data:
                    if isinstance(item, dict):
                        word_timestamps.append({
                            'word': item.get('word', ''),
                            'start': float(item.get('start', 0)),
                            'end': float(item.get('end', 0))
                        })
    except Exception as e:
        logger.warning(f"Failed to extract word timestamps: {e}")
    
    return word_timestamps


def extract_segment_timestamps(hypothesis) -> List[Dict[str, Any]]:
    """
    Extract segment-level timestamps from TDT model output
    
    Args:
        hypothesis: Hypothesis object from model.transcribe()
    
    Returns:
        List of segment timestamp dictionaries with 'text', 'start', 'end', 'word_count'
    """
    segment_timestamps = []
    
    try:
        # TDT model provides segment timestamps with populated text
        if hasattr(hypothesis, 'timestamp') and 'segment' in hypothesis.timestamp:
            segment_data = hypothesis.timestamp['segment']
            if isinstance(segment_data, list):
                for item in segment_data:
                    if isinstance(item, dict):
                        segment_text = item.get('segment', '')
                        # Count words in segment text for word_count field
                        word_count = len(segment_text.split()) if segment_text else 0
                        segment_timestamps.append({
                            'text': segment_text,
                            'start': float(item.get('start', 0)),
                            'end': float(item.get('end', 0)),
                            'word_count': word_count
                        })
    except Exception as e:
        logger.warning(f"Failed to extract segment timestamps: {e}")
    
    return segment_timestamps


# Pydantic models
class WordTimestamp(BaseModel):
    word: str
    start: float  # Start time in seconds
    end: float    # End time in seconds


class SegmentTimestamp(BaseModel):
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds
    word_count: int  # Number of words in segment


class TranscriptionMetadata(BaseModel):
    total_segments: int
    total_words: int
    duration: float


class TranscribeRequest(BaseModel):
    audio_url: HttpUrl


class TranscribeResponse(BaseModel):
    transcription: str
    processing_time: float
    word_timestamps: List[WordTimestamp]
    segment_timestamps: List[SegmentTimestamp]
    metadata: TranscriptionMetadata


# Model loading
@app.on_event("startup")
async def load_model():
    """Load the Parakeet TDT model on startup"""
    global model
    logger.info("Loading Parakeet TDT model...")
    try:
        # Set CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES
        
        # TDT models use generic ASRModel class
        model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.MODEL_NAME
        )
        
        # WORKAROUND: Disable CUDA graph decoder due to NeMo 2.6.x incompatibility
        # with CUDA 12.8+ (cudaStreamGetCaptureInfo API changed its return signature).
        # See: https://github.com/NVIDIA-NeMo/NeMo/issues/15145
        # TODO: Re-enable once NeMo releases a fix for this issue.
        from omegaconf import open_dict
        decoding_cfg = model.cfg.decoding
        with open_dict(decoding_cfg):
            decoding_cfg.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding_cfg)

        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


@app.on_event("shutdown")
async def cleanup():
    """Cleanup on shutdown"""
    logger.info("Shutting down service...")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "Parakeet TDT ASR Service",
        "version": "3.0.0"
    }


# Transcription endpoint
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(request: TranscribeRequest):
    """
    Transcribe audio file from URL with automatic word and segment-level timestamps
    
    Args:
        request: JSON with audio_url field
        
    Returns:
        Transcription text, word timestamps, segment timestamps, processing time, and metadata
    """
    start_time = time.time()
    temp_path = None
    preprocessed_path = None
    
    try:
        # Download file from URL
        audio_url_str = str(request.audio_url)
        logger.info(f"Downloading audio from URL: {audio_url_str}")
        
        # Fix potential double-encoding issues by normalizing the URL
        # Handle malformed URLs where %% appears (client encoding issue)
        if '%%' in audio_url_str:
            logger.warning(f"Detected malformed URL with %%, attempting to fix")
            # Parse URL to get the path component
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(audio_url_str)
            # Decode the path completely, then re-encode properly
            decoded_path = unquote(parsed.path)
            # Re-encode properly
            fixed_path = quote(decoded_path, safe='/')
            # Rebuild URL
            audio_url_str = urlunparse((parsed.scheme, parsed.netloc, fixed_path, '', '', ''))
            logger.info(f"Fixed URL: {audio_url_str}")
        
        logger.info(f"Initiating HTTP GET request...")
        response = requests.get(audio_url_str, timeout=600, stream=True, allow_redirects=True)  # 10 min timeout
        logger.info(f"HTTP response status: {response.status_code}, content-type: {response.headers.get('content-type', 'N/A')}, content-length: {response.headers.get('content-length', 'N/A')}")
        response.raise_for_status()
        logger.info("HTTP response OK, proceeding to parse file extension...")
        
        # Get file extension from URL or content-type
        url_path = Path(request.audio_url.path)
        file_ext = url_path.suffix.lower()
        logger.info(f"Detected file extension: '{file_ext}' from URL path: '{request.audio_url.path}'")
        
        if not file_ext or file_ext not in config.ALLOWED_EXTENSIONS:
            # Try to infer from content-type
            content_type = response.headers.get('content-type', '')
            if 'wav' in content_type:
                file_ext = ".wav"
            elif 'mp3' in content_type or 'mpeg' in content_type:
                file_ext = ".mp3"
            elif 'flac' in content_type:
                file_ext = ".flac"
            elif 'ogg' in content_type:
                file_ext = ".ogg"
            elif 'm4a' in content_type or 'mp4' in content_type:
                file_ext = ".m4a"
            else:
                file_ext = ".wav"  # Default to wav
        
        # Create temp file
        temp_path = os.path.join(config.TEMP_DIR, f"{int(time.time())}_download{file_ext}")
        
        # Save downloaded content
        logger.info(f"Saving downloaded content to: {temp_path}")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        logger.info(f"Downloaded file size: {file_size / (1024*1024):.2f}MB (limit: {config.MAX_FILE_SIZE_MB}MB)")
        if file_size > config.MAX_FILE_SIZE:
            logger.error(f"File too large: {file_size / (1024*1024):.2f}MB exceeds {config.MAX_FILE_SIZE_MB}MB limit")
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {config.MAX_FILE_SIZE_MB}MB"
            )
        
        # Preprocess audio (convert to mono, resample to 16kHz)
        logger.info("Starting audio preprocessing...")
        preprocessed_path = preprocess_audio(temp_path)
        logger.info(f"Audio preprocessed successfully: {preprocessed_path}")
        
        # Check if file should be processed in chunks
        if file_size > config.CHUNK_THRESHOLD:
            logger.info(f"Large file detected ({file_size / (1024*1024):.2f}MB), processing in chunks")
            
            # Split into chunks
            chunks = split_audio_into_chunks(
                preprocessed_path,
                chunk_duration=config.CHUNK_DURATION,
                overlap=config.CHUNK_OVERLAP
            )
            
            # Process each chunk sequentially
            chunk_results = []
            chunk_paths_to_cleanup = []
            
            try:
                for idx, chunk_info in enumerate(chunks):
                    logger.info(f"Processing chunk {idx+1}/{len(chunks)}...")
                    chunk_result = process_single_chunk(
                        chunk_info['path'],
                        chunk_info['offset']
                    )
                    chunk_results.append(chunk_result)
                    
                    # Keep track of chunk files for cleanup
                    if chunk_info['path'] != preprocessed_path:
                        chunk_paths_to_cleanup.append(chunk_info['path'])
                
                # Merge all chunk results
                merged_result = merge_transcription_chunks(chunk_results)
                transcription_text = merged_result['transcription']
                word_timestamps_raw = merged_result['word_timestamps']
                segment_timestamps_raw = merged_result['segment_timestamps']
                
            finally:
                # Cleanup chunk files
                for chunk_path in chunk_paths_to_cleanup:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
                logger.info(f"Cleaned up {len(chunk_paths_to_cleanup)} chunk files")
        
        else:
            # Normal processing for small files
            logger.info(f"Processing file normally ({file_size / (1024*1024):.2f}MB)")
            
            # Transcribe with timestamps (always enabled)
            logger.info(f"Transcribing audio from URL: {request.audio_url}")
            transcriptions = model.transcribe([preprocessed_path], timestamps=True)
            
            # Extract text from Hypothesis object
            result = transcriptions[0]
            transcription_text = result.text if hasattr(result, 'text') else str(result)
            
            # Extract word-level timestamps from TDT model
            word_timestamps_raw = extract_word_timestamps(result)
            
            # Extract segment-level timestamps from TDT model (native, no post-processing needed)
            segment_timestamps_raw = extract_segment_timestamps(result)
        
        # Calculate metadata
        metadata_raw = calculate_metadata(
            transcription_text,
            word_timestamps_raw,
            segment_timestamps_raw
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Transcription completed in {processing_time:.2f}s")
        logger.info(f"Generated {len(word_timestamps_raw)} words, {len(segment_timestamps_raw)} segments")
        
        # Convert to Pydantic models for response
        word_timestamps_models = [WordTimestamp(**wt) for wt in word_timestamps_raw]
        segment_timestamps_models = [SegmentTimestamp(**st) for st in segment_timestamps_raw]
        metadata_model = TranscriptionMetadata(**metadata_raw)
        
        # Create response object
        response_data = TranscribeResponse(
            transcription=transcription_text,
            processing_time=round(processing_time, 2),
            word_timestamps=word_timestamps_models,
            segment_timestamps=segment_timestamps_models,
            metadata=metadata_model
        )
        
        # Save transcription to local file
        try:
            transcripts_dir = Path(config.BASE_DIR) / "transcripts"
            transcripts_dir.mkdir(exist_ok=True)
            
            # Create filename from timestamp and URL
            url_filename = Path(request.audio_url.path).stem  # Get filename without extension
            timestamp = int(start_time)
            output_filename = f"{timestamp}_{url_filename}.json"
            output_path = transcripts_dir / output_filename
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(response_data.dict(), f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transcription saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save transcription to file: {e}")
            # Don't fail the request if file saving fails
        
        return response_data
        
    except requests.RequestException as e:
        logger.error(f"Failed to download audio: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio: {str(e)}")
    except HTTPException as e:
        logger.error(f"HTTPException raised: status_code={e.status_code}, detail={e.detail}")
        raise
    except Exception as e:
        logger.error(f"Transcription failed ({type(e).__name__}): {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        if preprocessed_path and preprocessed_path != temp_path and os.path.exists(preprocessed_path):
            os.remove(preprocessed_path)
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
