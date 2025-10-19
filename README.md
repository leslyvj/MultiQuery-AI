# Multimodal RAG System

A production-ready multimodal Retrieval-Augmented Generation (RAG) system that supports documents, images, and audio files with ChatGPT-style formatted answers.

## Features

- **Multi-Format Support**: PDF, DOCX, TXT, Images (PNG, JPG), Audio (MP3, WAV, M4A, FLAC)
- **Advanced Search**: Text queries, image queries, and audio queries
- **OCR Integration**: Extracts text from images using EasyOCR
- **Audio Transcription**: Faster-Whisper for GPU-accelerated transcription
- **ChatGPT-Style Answers**: Clean, scannable, structured responses
- **Offline LLM**: Phi-3 model via Ollama (no internet required)
- **Modern UI**: Clean web interface with statistics dashboard

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Ollama and Phi-3

```bash
# Install Ollama from https://ollama.ai
ollama pull phi3
```

### 3. Start the Server

**Option A - Double-click:**
- Run `START_HERE.bat`

**Option B - Command line:**
```bash
python run_server.py
```

### 4. Access the Interface

Open your browser to: `http://127.0.0.1:8000`

## Usage

### Upload Files
1. Click "Choose Files" in the Upload section
2. Select your documents, images, or audio files
3. Click "Upload & Index"
4. Wait for processing to complete

### Query Your Data
**Text Query:**
- Type your question in the text box
- Click "Search & Answer"

**Image Query:**
- Upload an image with text
- System will OCR the text and search for related content

**Audio Query:**
- Upload an audio file
- System will transcribe and search

## Project Structure

```
multimodal_rag_free/
├── app.py              # Main FastAPI server
├── run_server.py       # Server launcher
├── embeddings.py       # Text embedding engine
├── clip_embeddings.py  # Image embedding engine
├── ocr_engine.py       # OCR processing
├── ingestion.py        # File processing pipeline
├── indexer.py          # ChromaDB vector store
├── generator.py        # LLM backends
├── llama_query.py      # Answer generation
├── utils.py            # Prompt engineering
├── format_answer.py    # Answer formatting
├── web/                # Frontend files
├── chroma_db/          # Vector database
├── uploads/            # Uploaded files
└── requirements.txt    # Python dependencies
```

## Requirements

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA (recommended for audio transcription)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for models and dependencies

## Configuration

Edit `.env` file to customize:
```
EMBED_DEVICE=cuda          # or 'cpu'
LLAMA_MODEL_PATH=phi3      # Ollama model name
CHUNK_SIZE=500             # Text chunk size
CHUNK_OVERLAP=50           # Chunk overlap
TOP_K=5                    # Number of results
```

## Troubleshooting

**Server won't start:**
- Check if port 8000 is available
- Verify Ollama is running: `ollama list`
- Check `.env` configuration

**Slow audio processing:**
- Ensure GPU drivers are installed
- Set `EMBED_DEVICE=cuda` in `.env`

**OCR not working:**
- EasyOCR will download models on first use
- Requires ~500MB storage

## Support

For issues or questions, check the server logs in the terminal.

## License

MIT License - See LICENSE file for details
