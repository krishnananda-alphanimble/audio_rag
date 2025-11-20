import os
import tempfile
import shutil
from google import genai
from dotenv import load_dotenv
from audio_utils import get_audio_duration, slice_audio_interval, generate_intervals

load_dotenv()

# Ensure GOOGLE_API_KEY is set, though genai.Client() might pick it up automatically if set in env.
# We'll explicitly check for it to be safe/informative.
if not os.getenv("GOOGLE_API_KEY"):
    print("Warning: GOOGLE_API_KEY not found in environment. Gemini captioner might fail if not authenticated.")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

CAPTION_PROMPT = """
Analyze the audio track and generate a granular, verbose description of all non-speech audio events. Do not use standard concise captioning. Instead, deconstruct every sound effect and musical cue into a detailed narrative paragraph. For every sound, describe: (1) The physical source and material properties (e.g., 'heavy boots crushing dry autumn leaves on concrete'); (2) The acoustic characteristics (timbre, pitch, reverb, decay); (3) The spatial environment (e.g., 'muffled sound coming from behind a wall,' or 'echoing in a large metallic hangar'); and (4) The emotional or narrative function (e.g., 'creates a sense of imminent danger'). If multiple sounds occur simultaneously, describe the layering of the soundscape, distinguishing between foreground focus and background ambience. For music, describe specific instrumentation, tempo changes, and complex harmonic moods..
"""


# Audio utility functions moved to audio_utils.py


def generate_audio_caption_gemini(audio_path: str, model: str = "gemini-2.5-flash", temporal_context: str = None) -> str:
    """
    Generate a detailed audio caption using Gemini 2.5 Flash.
    
    Args:
        audio_path: Path to the audio file
        model: Gemini model to use
        temporal_context: Optional context about when this audio occurs (e.g., "This audio segment starts at 60s")
    """
    print(f"\n[2/5] Generating audio caption with Gemini ({model})...")
    
    try:
        # Upload the file
        print(f"  Uploading {os.path.basename(audio_path)} to Gemini...")
        myfile = client.files.upload(file=audio_path)
        
        # Build prompt with temporal context if provided
        prompt = CAPTION_PROMPT
        if temporal_context:
            prompt = f"{temporal_context}\n\n{CAPTION_PROMPT}"
        
        # Generate content
        print("  Generating content...")
        response = client.models.generate_content(
            model=model,
            contents=[prompt, myfile]
        )
        
        caption = response.text
        print(f"✓ Caption: {caption[:100]}...")
        return caption
        
    except Exception as e:
        print(f"Error generating caption with Gemini: {e}")
        raise e


def generate_interval_captions_gemini(
    audio_path: str,
    interval_duration: int = 60,
    padding: int = 5,
    model: str = "gemini-2.5-flash"
) -> dict:
    """
    Generate captions for non-overlapping intervals of an audio file.
    
    Intervals are simple 60s segments (0-60s, 60-120s, etc.). When slicing audio,
    padding is added before/after for context (e.g., slice 0-65s for interval 0-60s).
    
    Args:
        audio_path: Path to the audio file
        interval_duration: Duration of each interval in seconds (default: 60)
        padding: Seconds to pad before/after when slicing audio (default: 5)
        model: Gemini model to use (default: "gemini-2.5-flash")
    
    Returns:
        Dictionary mapping interval start times to caption data:
        {
            0: {"start": 0, "end": 60, "caption": "..."},
            60: {"start": 60, "end": 120, "caption": "..."},
            ...
        }
    """
    print(f"\n[2/5] Generating interval captions ({interval_duration}s intervals with {padding}s padding)...")
    
    # Get total audio duration
    duration = get_audio_duration(audio_path)
    print(f"  Audio duration: {duration:.2f}s")
    
    # Calculate non-overlapping intervals
    intervals = generate_intervals(duration, interval_duration)
    print(f"  Generated {len(intervals)} intervals")
    
    # Generate captions for each interval
    interval_captions = {}
    temp_dir = tempfile.mkdtemp()
    
    try:
        for i, interval in enumerate(intervals):
            print(f"\n  Processing interval {i+1}/{len(intervals)}: {interval['start']:.1f}s - {interval['end']:.1f}s")
            
            # Create temp file for sliced audio
            temp_audio_path = os.path.join(temp_dir, f"interval_{interval['key']}.wav")
            
            # Apply padding when slicing audio (not in interval definition)
            slice_start = max(0, interval["start"] - padding)
            slice_end = min(duration, interval["end"] + padding)
            
            # Slice audio with padding
            slice_audio_interval(
                audio_path,
                slice_start,
                slice_end,
                temp_audio_path
            )
            
            # Build temporal context for the model
            if interval["start"] == 0:
                temporal_context = f"Note: This audio segment is from the BEGINNING of the recording (starts at 0:00)."
            else:
                minutes = int(interval["start"] // 60)
                seconds = int(interval["start"] % 60)
                temporal_context = f"Note: This audio segment starts at {minutes}:{seconds:02d} into the recording, so it may begin mid-sentence or mid-thought."
            
            # Generate caption for this interval with temporal context
            caption = generate_audio_caption_gemini(temp_audio_path, model, temporal_context)
            
            # Store with key as interval start (for chunk mapping)
            interval_captions[interval["key"]] = {
                "start": interval["start"],
                "end": interval["end"],
                "caption": caption
            }
            
            print(f"    ✓ Caption generated: {caption[:80]}...")
    
    finally:
        # Clean up temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    print(f"\n✓ Generated {len(interval_captions)} interval captions")
    return interval_captions

