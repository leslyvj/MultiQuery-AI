# 🔍 Multimodal RAG System - SIH 2025

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)

**A production-ready Multimodal Retrieval-Augmented Generation (RAG) system for NTRO's Smart India Hackathon 2025**

> 🏆 **Built for:** National Technical Research Organisation (NTRO)  
> 🎯 **Category:** Software | Smart Automation  
> 💡 **Theme:** Offline Multimodal Intelligence with LLM Grounding

## 📋 Problem Statement

Design and build a multimodal RAG system that can **ingest, index, and query diverse data formats** (DOC, PDF, Images, Audio) within a **unified semantic retrieval framework** using a **Large Language Model in OFFLINE mode**.

## ✨ Key Features

### 🎯 Core Capabilities
- ✅ **Multimodal Ingestion**: PDF, DOCX, TXT, Images (PNG/JPG), Audio (MP3/WAV/M4A/FLAC)
- ✅ **OCR Integration**: EasyOCR with GPU acceleration for text extraction from images
- ✅ **Speech-to-Text**: Faster-Whisper for high-speed audio transcription
- ✅ **Vector Database**: ChromaDB for semantic search across all modalities
- ✅ **Offline LLM**: Phi-3 via Ollama (no internet required)
- ✅ **GPU Optimized**: Full CUDA acceleration (20-30% faster responses)
- ✅ **ChatGPT-Style UI**: Clean, professional web interface

### 🔎 Search Modes
1. **Text Query** - Natural language questions
2. **Image Query** - Upload images/screenshots, extract text via OCR, find related content
3. **Audio Query** - Upload voice recordings, transcribe, search semantically

### 📊 Advanced Features
- **Quality Scoring**: Relevance percentages for each source (0-100%)
- **Citation Tracking**: Numbered references with source file links
- **Cross-Format Search**: Text embeddings + CLIP visual embeddings
- **Source Navigation**: View original files, timestamps, and metadata
- **Statistics Dashboard**: Real-time metrics on indexed content

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for speed)
- 8GB RAM minimum (16GB recommended)
- 10GB free disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/multimodal_rag_free.git
cd multimodal_rag_free
```

2. **Create virtual environment:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install Ollama and Phi-3 model:**
```bash
# Download Ollama from https://ollama.ai
ollama pull phi3
```

5. **Start the server:**
```bash
# Option 1: Double-click START_HERE.bat (Windows)

# Option 2: Command line
python run_server.py

# Option 3: GPU-optimized startup
START_SERVER_GPU_OPTIMIZED.bat
```

6. **Open browser:**
Navigate to `http://127.0.0.1:8000`

## 📖 Usage Guide

### 1️⃣ Upload Documents
- Click "Choose Files" and select your documents, images, or audio files
- Multiple file types supported in one upload
- System automatically processes and indexes content

### 2️⃣ Query Your Data

**Text Mode:**
```
Example: "How to make one pan chicken?"
```

**Image Mode:**
- Upload an image or screenshot
- System extracts text via OCR
- Searches for related documents and images

**Audio Mode:**
- Upload voice recording or audio file
- System transcribes speech
- Searches for matching content

### 3️⃣ View Results
- **Answer**: Structured response with Summary, Key Points, Steps, Insights, Takeaway
- **Sources**: Numbered citations with relevance scores
- **View Source**: Click to open original files

## 🏗️ Project Structure

```
multimodal_rag_free/
├── app.py                 # FastAPI server with endpoints
├── run_server.py          # Server launcher
├── generator.py           # Multi-backend LLM generator
├── llama_query.py         # Answer generation orchestrator
├── embeddings.py          # Text embedding engine (sentence-transformers)
├── clip_embeddings.py     # Image embedding engine (CLIP)
├── ocr_engine.py          # OCR processing (EasyOCR)
├── ingestion.py           # Document processing pipeline
├── indexer.py             # ChromaDB vector store manager
├── utils.py               # Prompt engineering templates
├── format_answer.py       # Answer post-processing
├── web/                   # Frontend interface
│   └── index.html         # Modern single-page UI
├── chroma_db/             # Vector database (gitignored)
├── uploads/               # Uploaded files (gitignored)
├── requirements.txt       # Python dependencies
├── .env                   # Configuration file
└── START_HERE.bat         # Quick launcher
```

## ⚙️ Configuration

Edit `.env` to customize:

```properties
# Database
CHROMA_PERSIST_DIR=./chroma_db

# LLM Model
LLAMA_MODEL_PATH=phi3
LLAMA_MODEL_ID=phi3

# Device
EMBED_DEVICE=cuda  # or 'cpu'

# Audio
WHISPER_MODEL=base

# Server
PORT=8000
```

## 🎯 SIH 2025 Requirements Compliance

| Requirement | Implementation | Status |
|------------|----------------|--------|
| Multimodal Ingestion | PDF, DOCX, TXT, Images, Audio | ✅ |
| OCR for Images | EasyOCR with GPU | ✅ |
| Speech-to-Text | Faster-Whisper | ✅ |
| Vector Indexing | ChromaDB + CLIP | ✅ |
| Semantic Search | Unified vector space | ✅ |
| LLM Generation | Phi-3 (Offline) | ✅ |
| Natural Language Query | Plain text interface | ✅ |
| Citation Transparency | Numbered references | ✅ |
| Source Navigation | File viewing | ✅ |
| GPU Acceleration | CUDA optimized | ✅ |

## 🔧 GPU Optimization

For maximum performance with NVIDIA GPUs:

```bash
# Option 1: Use optimized startup script
START_SERVER_GPU_OPTIMIZED.bat

# Option 2: Set environment variables manually
$env:OLLAMA_NUM_GPU=999
$env:OLLAMA_MAX_LOADED_MODELS=1
$env:OLLAMA_FLASH_ATTENTION=1
python run_server.py
```

**Expected Improvements:**
- 20-30% faster LLM generation
- Lower latency between queries
- Model stays in GPU memory (no reload delay)

## 📊 Performance Metrics

- **Response Time**: 5-7 seconds per query (GPU optimized)
- **Accuracy**: 85-95% relevance with quality scoring
- **Throughput**: Handles 100+ documents, images, audio files
- **Memory**: ~4GB GPU VRAM, ~8GB system RAM

## 🛠️ Troubleshooting

### Server Won't Start
```bash
# Check if port is available
netstat -ano | findstr :8000

# Verify Ollama is running
ollama list
```

### Slow Performance
```bash
# Use GPU-optimized startup
START_SERVER_GPU_OPTIMIZED.bat

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### OCR Not Working
- EasyOCR downloads models on first use (~500MB)
- Ensure stable internet for initial setup
- Check GPU drivers are installed

## 🤝 Contributing

This project is part of SIH 2025. For contributions:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Organization:** National Technical Research Organisation (NTRO)  
**Theme:** Smart Automation  
**Year:** 2025

## 🙏 Acknowledgments

- **Ollama** - Offline LLM deployment
- **ChromaDB** - Vector database
- **OpenAI CLIP** - Visual embeddings
- **EasyOCR** - Text extraction
- **Faster-Whisper** - Audio transcription

## 📧 Contact

For questions or support, please open an issue in the GitHub repository.

---

**Built with ❤️ for Smart India Hackathon 2025**
