import os
import base64
import whisper
from dotenv import load_dotenv
from mistralai import Mistral
import chromadb
import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from midas_captioner import generate_audio_caption_midas, generate_interval_captions_midas
from mistral_captioner import generate_audio_caption_mistral, generate_interval_captions_mistral
from gemini_captioner import generate_audio_caption_gemini, generate_interval_captions_gemini
from audio_utils import get_audio_duration, slice_audio_interval, generate_intervals
import tempfile
import shutil

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in environment")

mistral_client = Mistral(api_key=MISTRAL_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def transcribe_audio(audio_path: str, model_size: str = "tiny") -> list:
    print(f"\n[1/4] Transcribing audio with Whisper ({model_size})...")
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, language="en", word_timestamps=True)
    
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip()
        })
    
    print(f"✓ Transcription complete: {len(segments)} segments")
    return segments




def generate_audio_caption_qwen(
    audio_path: str,
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    prompt: str = "Transcribe the audio.",
    use_audio_in_video: bool = False,
    max_new_tokens: int = 256,
    device: str = None,
    temporal_context: str = None
) -> str:
    """
    Generate an audio caption using Qwen3-Omni model.
    
    Args:
        audio_path: Path to the audio file
        model_path: HuggingFace model path (default: "Qwen/Qwen3-Omni-30B-A3B-Captioner")
        prompt: Text prompt for captioning (default: "Transcribe the audio.")
        use_audio_in_video: Whether to use audio in video (default: False)
        max_new_tokens: Maximum tokens to generate (default: 256)
        device: Device to run on (default: auto-detect)
        temporal_context: Optional context about when this audio occurs
    
    Returns:
        Generated caption as a string
    """
    print(f"\n[2/5] Generating audio caption with Qwen3-Omni...")
    
    # Auto-detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Qwen3-Omni model on {device}...")
    
    # Load model and processor
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
    
    # Build prompt with temporal context if provided
    final_prompt = prompt
    if temporal_context:
        final_prompt = f"{temporal_context}\n\n{prompt}"
    
    # Define conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": final_prompt},
            ],
        }
    ]
    
    # Preprocess
    print("Preprocessing audio input...")
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=use_audio_in_video,
    )
    inputs = inputs.to(model.device).to(model.dtype)
    
    # Generate caption
    print("Generating caption...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            speaker="Ethan",
            use_audio_in_video=use_audio_in_video,
            return_dict_in_generate=False,
            max_new_tokens=max_new_tokens,
        )
    
    # Decode
    caption = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    caption_text = caption[0] if caption else ""
    
    print(f"✓ Caption: {caption_text[:100]}...")
    return caption_text


def generate_interval_captions_qwen(
    audio_path: str,
    interval_duration: int = 60,
    padding: int = 5,
    model_path: str = "Qwen/Qwen3-Omni-30B-A3B-Captioner",
    prompt: str = "Transcribe the audio.",
    use_audio_in_video: bool = False,
    max_new_tokens: int = 256,
    device: str = None
) -> dict:
    """
    Generate captions for non-overlapping intervals of an audio file using Qwen3-Omni.
    
    Intervals are simple 60s segments (0-60s, 60-120s, etc.). When slicing audio,
    padding is added before/after for context (e.g., slice 0-65s for interval 0-60s).
    
    Args:
        audio_path: Path to the audio file
        interval_duration: Duration of each interval in seconds (default: 60)
        padding: Seconds to pad before/after when slicing audio (default: 5)
        model_path: HuggingFace model path
        prompt: Text prompt for captioning
        use_audio_in_video: Whether to use audio in video
        max_new_tokens: Maximum tokens to generate
        device: Device to run on (default: auto-detect)
    
    Returns:
        Dictionary mapping interval start times to caption data:
        {
            0: {"start": 0, "end": 60, "caption": "..."},
            60: {"start": 60, "end": 120, "caption": "..."},
            ...
        }
    """
    print(f"\n[2/5] Generating interval captions with Qwen ({interval_duration}s intervals with {padding}s padding)...")
    
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
            caption = generate_audio_caption_qwen(
                temp_audio_path, 
                model_path, 
                prompt, 
                use_audio_in_video, 
                max_new_tokens, 
                device,
                temporal_context
            )
            
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




def format_timestamp(start: float, end: float) -> str:
    def sec_to_mmss(sec):
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"
    
    return f"{sec_to_mmss(start)} → {sec_to_mmss(end)}"


def create_chunks(segments: list, chunk_duration: int = 20, overlap: int = 2) -> list:
    print(f"\n[3/5] Creating chunks (duration={chunk_duration}s, overlap={overlap}s)...")
    
    chunks = []
    current_segments = []
    chunk_start = segments[0]["start"] if segments else 0
    
    for seg in segments:
        if seg["end"] - chunk_start > chunk_duration and current_segments:
            chunk_end = current_segments[-1]["end"]
            chunk_text = " ".join(s["text"] for s in current_segments)
            
            chunks.append({
                "start": chunk_start,
                "end": chunk_end,
                "text": chunk_text
            })
            
            overlap_start = max(0, chunk_end - overlap)
            current_segments = [s for s in current_segments if s["end"] > overlap_start]
            chunk_start = current_segments[0]["start"] if current_segments else seg["start"]
        
        current_segments.append(seg)
    
    if current_segments:
        chunk_end = current_segments[-1]["end"]
        chunk_text = " ".join(s["text"] for s in current_segments)
        chunks.append({
            "start": chunk_start,
            "end": chunk_end,
            "text": chunk_text
        })
    
    print(f"✓ Created {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks: list, interval_captions: dict) -> list:
    """
    Generate embeddings for chunks. Each chunk is enriched with the caption
    from the appropriate 60s interval based on its start time.
    
    Args:
        chunks: List of transcript chunks with start/end times
        interval_captions: Dictionary mapping interval start times to caption data
    
    Returns:
        List of enriched chunks with embeddings
    """
    print(f"\n[4/5] Generating embeddings for {len(chunks)} chunks...")
    
    enriched = []
    for i, chunk in enumerate(chunks):
        # Find the appropriate interval caption based on chunk start time
        # Interval key is the floor of (start_time / 60) * 60
        interval_key = (int(chunk['start']) // 60) * 60
        
        # Get the caption for this interval
        if interval_key in interval_captions:
            audio_caption = interval_captions[interval_key]["caption"]
        else:
            # Fallback: use the last available interval if chunk is beyond calculated intervals
            max_key = max(interval_captions.keys())
            audio_caption = interval_captions[max_key]["caption"]
            print(f"  Warning: Chunk {i} at {chunk['start']}s mapped to fallback interval {max_key}")
        
        # Build context with caption and transcript separate
        full_context = f"[Audio Context: {audio_caption}] [Transcript {format_timestamp(chunk['start'], chunk['end'])}]: {chunk['text']}"
        
        embedding = mistral_client.embeddings.create(
            model="mistral-embed",
            inputs=full_context
        ).data[0].embedding
        
        enriched.append({
            "id": f"chunk_{i}",
            "start": chunk["start"],
            "end": chunk["end"],
            "timestamp": format_timestamp(chunk["start"], chunk["end"]),
            "transcript": chunk["text"],
            "full_context": full_context,
            "embedding": embedding,
            "interval_key": interval_key,  # Store which interval was used
            "interval_caption": audio_caption  # Store the specific caption
        })
        
        print(f"  ✓ Chunk {i+1}/{len(chunks)} (interval {interval_key}s)")
    
    print(f"✓ All chunks embedded")
    return enriched


def store_in_chroma(enriched_chunks: list, interval_captions: dict, collection_name: str = "audio_rag"):
    """
    Store enriched chunks in ChromaDB with interval-specific caption metadata.
    
    Args:
        enriched_chunks: List of chunks with embeddings and interval captions
        interval_captions: Dictionary of all interval captions (for reference)
        collection_name: Name of the ChromaDB collection
    """
    print(f"\n[5/5] Storing {len(enriched_chunks)} chunks in Chroma...")
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        collection.delete(where={})
    except:
        pass
    
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    ids = [chunk["id"] for chunk in enriched_chunks]
    documents = [chunk["full_context"] for chunk in enriched_chunks]
    embeddings = [chunk["embedding"] for chunk in enriched_chunks]
    
    metadatas = [
        {
            "start": chunk["start"],
            "end": chunk["end"],
            "timestamp": chunk["timestamp"],
            "transcript": chunk["transcript"],
            "interval_key": chunk["interval_key"],
            "interval_caption": chunk["interval_caption"]
        }
        for chunk in enriched_chunks
    ]
    
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    print(f"✓ Stored in Chroma DB")


def build_audio_rag(audio_path: str, chunk_duration: int = 20, overlap: int = 2, captioner: str = "gemini"):
    """
    Build an audio RAG system with interval-based captioning.
    
    Args:
        audio_path: Path to the audio file
        chunk_duration: Duration of each transcript chunk in seconds (default: 20)
        overlap: Overlap between chunks in seconds (default: 2)
        captioner: Which captioner to use - "gemini", "mistral", "midas", or "qwen" (default: "gemini")
    
    Returns:
        Tuple of (enriched_chunks, interval_captions)
    """
    print("=" * 60)
    print("BUILDING AUDIO RAG")
    print("=" * 60)
    print(f"Captioner: {captioner.upper()}")
    
    # Step 1: Transcribe audio
    segments = transcribe_audio(audio_path)
    
    # Step 2: Generate interval captions (70s windows: 60s + 5s before + 5s after)
    # Choose captioner based on parameter
    if captioner.lower() == "gemini":
        interval_captions = generate_interval_captions_gemini(audio_path)
    elif captioner.lower() == "mistral":
        interval_captions = generate_interval_captions_mistral(audio_path)
    elif captioner.lower() == "midas":
        interval_captions = generate_interval_captions_midas(audio_path)
    elif captioner.lower() == "qwen":
        interval_captions = generate_interval_captions_qwen(audio_path)
    else:
        raise ValueError(f"Unknown captioner: {captioner}. Choose from: gemini, mistral, midas, qwen")
    
    # Step 3: Create chunks from transcript
    chunks = create_chunks(segments, chunk_duration, overlap)
    
    # Step 4: Embed chunks with interval-specific captions
    enriched_chunks = embed_chunks(chunks, interval_captions)
    
    # Step 5: Store in Chroma with interval caption metadata
    store_in_chroma(enriched_chunks, interval_captions)
    
    print("\n" + "=" * 60)
    print("✓ RAG BUILD COMPLETE")
    print("=" * 60)
    print(f"Captioner: {captioner.upper()}")
    print(f"Generated {len(interval_captions)} interval captions (70s windows)")
    print(f"Total chunks: {len(enriched_chunks)}")
    
    # Print interval summary
    print("\nInterval Summary:")
    for key in sorted(interval_captions.keys()):
        interval = interval_captions[key]
        print(f"  {key}s-{key+60}s (window: {interval['start']:.1f}s-{interval['end']:.1f}s)")
    
    print()
    return enriched_chunks, interval_captions


if __name__ == "__main__":
    AUDIO_FILE = "harvard.wav"
    # Choose captioner: "gemini", "mistral", "midas", or "qwen"
    CAPTIONER = "gemini"
    build_audio_rag(AUDIO_FILE, captioner=CAPTIONER)

