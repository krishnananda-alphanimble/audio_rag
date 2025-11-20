"""
Shared audio utility functions for audio processing.
Used by all captioners for slicing and duration extraction.
"""
import os
import soundfile as sf
import numpy as np


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds using soundfile.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Duration in seconds as a float
    """
    try:
        info = sf.info(audio_path)
        duration = info.duration
        print(f"Audio duration: {duration:.2f}s")
        return duration
    except Exception as e:
        raise RuntimeError(f"Failed to get audio duration: {e}")


def slice_audio_interval(audio_path: str, start_time: float, end_time: float, output_path: str) -> str:
    """
    Slice an audio file to extract a specific time interval using soundfile.
    
    Args:
        audio_path: Path to the source audio file
        start_time: Start time in seconds (will be clamped to 0 if negative)
        end_time: End time in seconds
        output_path: Path where the sliced audio will be saved
    
    Returns:
        Path to the output file
    """
    # Clamp start_time to 0
    start_time = max(0, start_time)
    
    # Calculate duration
    duration = end_time - start_time
    
    if duration <= 0:
        raise ValueError(f"Invalid interval: start={start_time}, end={end_time}")
    
    try:
        # Read audio file info
        info = sf.info(audio_path)
        sample_rate = info.samplerate
        
        # Calculate frame positions
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        
        # Clamp end_frame to file length
        total_frames = info.frames
        end_frame = min(end_frame, total_frames)
        
        # Read the specific segment
        data, sr = sf.read(
            audio_path,
            start=start_frame,
            stop=end_frame,
            dtype='float32'
        )
        
        # Write the sliced audio
        sf.write(output_path, data, sr)
        
        return output_path
    except Exception as e:
        raise RuntimeError(f"Failed to slice audio: {e}")


def generate_intervals(duration: float, interval_duration: int = 60) -> list:
    """
    Generate non-overlapping interval specifications for audio captioning.
    
    Intervals are simple sequential segments (0-60s, 60-120s, 120-180s, etc.).
    Padding should be applied during audio slicing, NOT in interval definition.
    
    Args:
        duration: Total audio duration in seconds
        interval_duration: Duration of each interval (default: 60s)
    
    Returns:
        List of interval dictionaries with keys: 'key', 'start', 'end'
        Example for 180s audio: [
            {"key": 0, "start": 0, "end": 60},
            {"key": 60, "start": 60, "end": 120},
            {"key": 120, "start": 120, "end": 180}
        ]
    """
    intervals = []
    current_start = 0
    
    while current_start < duration:
        # Simple non-overlapping intervals
        window_end = min(duration, current_start + interval_duration)
        
        intervals.append({
            "key": current_start,  # Key for mapping chunks (0, 60, 120, ...)
            "start": current_start,
            "end": window_end
        })
        
        current_start += interval_duration
    
    return intervals

