"""
MidasHengLM Audio Captioner
Uses mispeech/midashenglm-7b-0804-fp8 model for audio captioning.
"""

import os
import tempfile
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from audio_utils import get_audio_duration, slice_audio_interval, generate_intervals

caption_prompt = """
Analyze the audio track and generate a granular, verbose description of all non-speech audio events. Do not use standard concise captioning. Instead, deconstruct every sound effect and musical cue into a detailed narrative paragraph. For every sound, describe: (1) The physical source and material properties (e.g., 'heavy boots crushing dry autumn leaves on concrete'); (2) The acoustic characteristics (timbre, pitch, reverb, decay); (3) The spatial environment (e.g., 'muffled sound coming from behind a wall,' or 'echoing in a large metallic hangar'); and (4) The emotional or narrative function (e.g., 'creates a sense of imminent danger'). If multiple sounds occur simultaneously, describe the layering of the soundscape, distinguishing between foreground focus and background ambience. For music, describe specific instrumentation, tempo changes, and complex harmonic moods..
"""
def generate_audio_caption_midas(
    audio_path: str,
    prompt: str = caption_prompt,
    model_id: str = "mispeech/midashenglm-7b-0804-fp8",
    device: str = None,
    temporal_context: str = None
) -> str:
    """
    Generate an audio caption using the MidasHengLM model.
    
    Args:
        audio_path: Path to the audio file
        prompt: Text prompt for captioning (default: "Caption the audio.")
        model_id: HuggingFace model ID (default: "mispeech/midashenglm-7b-0804-fp8")
        device: Device to run the model on (default: auto-detect)
        temporal_context: Optional context about when this audio occurs
    
    Returns:
        Generated caption as a string
    """
    print("\n" + "=" * 60)
    print("MIDAS CAPTIONER - Starting")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt}")
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n[1/5] Device detection: {device}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    
    # Load model, tokenizer, and processor
    print(f"\n[2/5] Loading model components...")
    print(f"      Loading model from HuggingFace...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    print(f"      ✓ Model loaded")
    
    print(f"      Moving model to {device}...")
    model = model.to(device)
    print(f"      ✓ Model on {device}")
    
    print(f"      Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"      ✓ Tokenizer loaded")
    
    print(f"      Loading processor...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    print(f"      ✓ Processor loaded")
    
    # Construct the conversation messages
    print(f"\n[3/5] Constructing conversation messages...")
    
    # Build prompt with temporal context if provided
    final_prompt = prompt
    if temporal_context:
        final_prompt = f"{temporal_context}\n\n{prompt}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": final_prompt},
                {"type": "audio", "path": audio_path},
            ],
        },
    ]
    print(f"      ✓ Messages constructed")
    
    print(f"\n[4/5] Processing audio and preparing inputs...")
    
    # Generate caption
    with torch.no_grad():
        print(f"      Applying chat template...")
        model_inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_special_tokens=True,
            return_dict=True,
        ).to(device=model.device, dtype=model.dtype)
        print(f"      ✓ Chat template applied")
        
        print(f"\n[5/5] Generating caption (this may take a while)...")
        generation = model.generate(**model_inputs)
        print(f"      ✓ Generation complete")
        
        print(f"      Decoding output...")
        output = tokenizer.batch_decode(generation, skip_special_tokens=True)
        print(f"      ✓ Decoding complete")
    
    caption = output[0] if output else ""
    
    print("\n" + "=" * 60)
    print("MIDAS CAPTIONER - Complete")
    print("=" * 60)
    print(f"Caption: {caption}")
    print("=" * 60 + "\n")
    
    return caption


def generate_interval_captions_midas(
    audio_path: str,
    interval_duration: int = 60,
    padding: int = 5,
    prompt: str = caption_prompt,
    model_id: str = "mispeech/midashenglm-7b-0804-fp8",
    device: str = None
) -> dict:
    """
    Generate captions for non-overlapping intervals of an audio file using MidasHengLM.
    
    Intervals are simple 60s segments (0-60s, 60-120s, etc.). When slicing audio,
    padding is added before/after for context (e.g., slice 0-65s for interval 0-60s).
    
    Args:
        audio_path: Path to the audio file
        interval_duration: Duration of each interval in seconds (default: 60)
        padding: Seconds to pad before/after when slicing audio (default: 5)
        prompt: Text prompt for captioning
        model_id: HuggingFace model ID (default: "mispeech/midashenglm-7b-0804-fp8")
        device: Device to run on (default: auto-detect)
    
    Returns:
        Dictionary mapping interval start times to caption data:
        {
            0: {"start": 0, "end": 60, "caption": "..."},
            60: {"start": 60, "end": 120, "caption": "..."},
            ...
        }
    """
    print(f"\n[2/5] Generating interval captions with Midas ({interval_duration}s intervals with {padding}s padding)...")
    
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
            caption = generate_audio_caption_midas(temp_audio_path, prompt, model_id, device, temporal_context)
            
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


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "harvard.wav"  # Default audio file
    
    print("=" * 60)
    print("MidasHengLM Audio Captioner")
    print("=" * 60)
    
    caption = generate_audio_caption_midas(audio_file)
    
    print("\n" + "=" * 60)
    print("CAPTION RESULT")
    print("=" * 60)
    print(f"Audio: {audio_file}")
    print(f"Caption: {caption}")
    print("=" * 60)
