import os
import base64
import whisper
from dotenv import load_dotenv
from mistralai import Mistral
import chromadb

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


def generate_audio_caption(audio_path: str, model: str = "voxtral-mini-latest") -> str:
    """
    Generate a holistic caption describing the audio's characteristics using Mistral Voxtral.
    This is called once per audio file, not per chunk.
    """
    print(f"\n[2/5] Generating audio caption with Voxtral...")
    
    # Encode the audio file in base64
    with open(audio_path, "rb") as f:
        content = f.read()
    audio_base64 = base64.b64encode(content).decode('utf-8')
    
    response = mistral_client.chat.complete(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": audio_base64,
                },
                {
                    "type": "text",
                    "text": "Describe this audio in detail, including tone, background sounds, and context."
                },
            ]
        }],
    )
    
    caption = response.choices[0].message.content.strip()
    print(f"✓ Caption: {caption[:100]}...")
    return caption


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


def embed_chunks(chunks: list, audio_caption: str) -> list:
    """
    Generate embeddings for chunks. The audio caption is provided as context
    but kept separate from the transcript.
    """
    print(f"\n[4/5] Generating embeddings for {len(chunks)} chunks...")
    
    enriched = []
    for i, chunk in enumerate(chunks):
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
            "embedding": embedding
        })
        
        print(f"  ✓ Chunk {i+1}/{len(chunks)}")
    
    print(f"✓ All chunks embedded")
    return enriched


def store_in_chroma(enriched_chunks: list, audio_caption: str, collection_name: str = "audio_rag"):
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
            "audio_caption": audio_caption
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


def build_audio_rag(audio_path: str, chunk_duration: int = 20, overlap: int = 2):
    print("=" * 60)
    print("BUILDING AUDIO RAG")
    print("=" * 60)
    
    # Step 1: Transcribe audio
    segments = transcribe_audio(audio_path)
    
    # Step 2: Generate single audio caption
    audio_caption = generate_audio_caption(audio_path)
    
    # Step 3: Create chunks from transcript
    chunks = create_chunks(segments, chunk_duration, overlap)
    
    # Step 4: Embed chunks with audio caption as context
    enriched_chunks = embed_chunks(chunks, audio_caption)
    
    # Step 5: Store in Chroma with caption metadata
    store_in_chroma(enriched_chunks, audio_caption)
    
    print("\n" + "=" * 60)
    print("✓ RAG BUILD COMPLETE")
    print("=" * 60)
    print(f"Audio Caption: {audio_caption}")
    print(f"Total chunks: {len(enriched_chunks)}\n")
    
    return enriched_chunks, audio_caption


if __name__ == "__main__":
    AUDIO_FILE = "harvard.wav"
    build_audio_rag(AUDIO_FILE)
