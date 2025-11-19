# Audio RAG

Ask questions about your audio files using AI.

## What It Does

Converts audio to searchable text, then lets you query it conversationally. Uses Whisper for transcription, Mistral for AI, and Chroma for storage.

## Setup

**1. Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Restart your terminal after installation.

**2. Install dependencies**
```bash
uv sync
```

**3. Set API key**
Create `.env` file:
```
MISTRAL_API_KEY=your_key_here
```
Get key from https://console.mistral.ai

**4. Add your audio**
Put your `.wav` file in this folder. Edit `build_rag.py` line 11:
```python
AUDIO_FILE = "your_file.wav"
```

## Usage

**Build database (first time only):**
```bash
uv run python build_rag.py
```
Takes 2-5 minutes depending on file size.

**Ask questions:**
```bash
uv run python query_rag.py
```
## Chunking — simple explanation

Chunking is how we split a long audio file into small, searchable pieces (called "chunks"). This makes searching and answering questions fast and accurate.

- Each chunk is about 20 seconds long. We add a small 2-second overlap between chunks so that information at chunk edges isn't missed.
- For each chunk we keep: its text, a short caption (summary), and a time range. These are stored in the database for fast lookup.
- When you ask a question, the system finds the most relevant chunks and uses them to build an answer — so the AI can focus only on useful parts of the audio.

Tip: shorter chunks help find exact moments quickly, but longer chunks give more context. 20s with 2s overlap is a good default.

## How It Works

1. **Transcribe**: Audio → text with timestamps (Whisper)
2. **Chunk**: Split into 20-second overlapping segments
3. **Caption**: AI generates summaries for audio (Mistral)
4. **Embed**: Convert to vectors for search (Mistral)
5. **Store**: Save in database (Chroma)
6. **Query**: Search chunks and generate answers (Mistral)

### Why Chunking?

Long audio is split into 20-second pieces with 2-second overlaps. This makes search faster and prevents missing info at boundaries.

Example for 1-minute audio:
- Chunk 1: 0-20s
- Chunk 2: 18-38s
- Chunk 3: 36-56s
- Chunk 4: 54-60s

## Troubleshooting

**"MISTRAL_API_KEY not found"**
→ Create `.env` file with your key

**"No such file"**
→ Update `AUDIO_FILE` in `build_rag.py`

**Dependency issues**
```bash
uv sync --reinstall
```

---

