"""
Segment Utilities
Metadata calculation and segment merging for transcription results
"""
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def merge_segment_boundaries(prev_segments: List[Dict[str, Any]], curr_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge segment boundaries between chunks if they were split mid-sentence
    
    Args:
        prev_segments: Segments from previous chunk
        curr_segments: Segments from current chunk
        
    Returns:
        Merged segment list (curr_segments with first segment potentially merged)
    """
    if not prev_segments or not curr_segments:
        return curr_segments
    
    last_prev = prev_segments[-1]
    first_curr = curr_segments[0]
    
    # Check if last segment of prev chunk ends without sentence-ending punctuation
    last_text = last_prev['text'].rstrip()
    ends_with_punctuation = last_text.endswith(('.', '!', '?'))
    
    # Check if first segment of current chunk starts without capital letter
    first_text = first_curr['text'].lstrip()
    starts_with_lowercase = first_text and first_text[0].islower()
    
    # If both conditions met, segments should be merged
    if not ends_with_punctuation and starts_with_lowercase:
        logger.info(f"Merging segments across chunk boundary: '{last_text[-30:]}...' + '...{first_text[:30]}'")
        
        # Create merged segment
        merged_segment = {
            'text': last_text + ' ' + first_text,
            'start': last_prev['start'],
            'end': first_curr['end'],
            'word_count': last_prev['word_count'] + first_curr['word_count']
        }
        
        # Update prev_segments in-place (modify the last one)
        prev_segments[-1] = merged_segment
        
        # Return curr_segments without the first one (it was merged)
        return curr_segments[1:]
    
    # No merge needed, return as-is
    return curr_segments


def calculate_metadata(
    transcription: str,
    word_timestamps: List[Dict[str, Any]],
    segment_timestamps: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate metadata about the transcription
    
    Args:
        transcription: Full transcription text
        word_timestamps: List of word timestamp objects
        segment_timestamps: List of segment timestamp objects
    
    Returns:
        Dictionary with metadata statistics
    """
    if not word_timestamps:
        return {
            "total_segments": len(segment_timestamps),
            "total_words": 0,
            "duration": 0.0
        }
    
    # Calculate duration from last word end time
    last_word = word_timestamps[-1]
    if isinstance(last_word, dict):
        duration = last_word.get('end', 0.0)
    else:
        duration = getattr(last_word, 'end', 0.0)
    
    return {
        "total_segments": len(segment_timestamps),
        "total_words": len(word_timestamps),
        "duration": round(duration, 2)
    }


