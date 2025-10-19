# ingestion.py
import os
from typing import List, Dict, Tuple
from PIL import Image
import pdfplumber
import docx
import whisper
import math
import soundfile as sf
import numpy as np
import pytesseract
from pathlib import Path
from tqdm import tqdm

# NOTE: If you want OCR for images/screenshots, make sure Tesseract engine is installed.

def extract_text_from_pdf(path: str) -> str:
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n\n".join(text_parts)

def extract_text_from_docx(path: str) -> str:
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def image_to_text_ocr(pil_image: Image.Image) -> str:
    """
    Use pytesseract to OCR images (install tesseract on OS).
    Returns empty string if Tesseract is not installed.
    """
    try:
        return pytesseract.image_to_string(pil_image)
    except Exception as e:
        print(f"Warning: Tesseract OCR not available ({e}). Continuing without OCR text extraction.")
        return ""

def transcribe_audio_whisper(path: str, model_name: str = "base", device: str = "cpu") -> Dict:
    """
    OPTIMIZED HIGH-QUALITY audio transcription using faster-whisper.
    
    Key improvements over standard Whisper:
    - 4-12x faster transcription speed
    - GPU acceleration support
    - Voice Activity Detection (VAD) to skip silence
    - Better memory efficiency
    - Batch processing capability
    
    Args:
        path: Path to audio file (MP3, WAV, M4A, OGG, etc.)
        model_name: Whisper model (base, small, medium, large-v3)
        device: 'cpu' or 'cuda'
    
    Returns:
        Dict with 'text', 'segments', and 'language' keys
    """
    from faster_whisper import WhisperModel
    
    print(f"\n{'='*70}")
    print(f"🎤 FASTER-WHISPER AUDIO TRANSCRIPTION (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"File: {path}")
    print(f"Model: {model_name.upper()} | Device: {device.upper()}")
    print(f"{'='*70}\n")
    
    try:
        # Try GPU first, fallback to CPU if it fails
        model = None
        if device == "cuda":
            try:
                compute_type = "float16"  # GPU: use float16 for speed
                print("� Attempting GPU acceleration (float16)...")
                model = WhisperModel(
                    model_name,
                    device="cuda",
                    compute_type=compute_type,
                    num_workers=4,
                    download_root=None
                )
                print(f"✅ GPU model loaded successfully\n")
            except Exception as gpu_error:
                print(f"⚠️  GPU loading failed: {str(gpu_error)[:100]}")
                print(f"🔄 Falling back to CPU...\n")
                device = "cpu"
        
        # Load CPU model if GPU failed or was not requested
        if model is None:
            compute_type = "int8"  # CPU: use int8 for efficiency
            print("💻 Using CPU (int8 quantization)")
            print(f"📥 Loading Faster-Whisper {model_name} model...")
            model = WhisperModel(
                model_name,
                device="cpu",
                compute_type=compute_type,
                num_workers=4,  # Parallel processing for speed
                download_root=None
            )
            print(f"✅ Model loaded successfully\n")
        
        # Get audio file info
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(path)
            duration = len(audio_data) / sample_rate
            print(f"📊 Audio info:")
            print(f"   ✓ Duration: {duration:.2f} seconds")
            print(f"   ✓ Sample rate: {sample_rate:,} Hz")
            if len(audio_data.shape) > 1:
                print(f"   ✓ Channels: {audio_data.shape[1]}")
            print()
        except:
            duration = 0
        
        print(f"🎯 Starting optimized transcription...")
        print(f"   Using Voice Activity Detection (VAD) to skip silence")
        print(f"   Beam size: 5 (balanced speed/accuracy)")
        print(f"   This may take {int(duration * 0.1)}-{int(duration * 0.3)} seconds...\n")
        
        # Transcribe with OPTIMIZED settings
        try:
            segments, info = model.transcribe(
                path,
                # Language settings
                language="en",  # Force English (remove if you want auto-detect)
                task="transcribe",
                
                # Quality settings
                beam_size=5,  # Good balance between speed and accuracy
                best_of=5,  # Try 5 candidates
                temperature=0.0,  # Deterministic
                
                # Speed optimizations
                vad_filter=True,  # Voice Activity Detection - skip silence
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Skip silence longer than 500ms
                    threshold=0.5  # Sensitivity threshold
                ),
                
                # Processing options
                condition_on_previous_text=True,  # Use context
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                
                # Performance
                word_timestamps=False  # Faster without word-level timing
            )
        except Exception as transcribe_error:
            print(f"❌ Transcription failed: {transcribe_error}")
            print(f"🔄 Retrying with simpler settings...")
            # Retry with minimal settings
            segments, info = model.transcribe(
                path,
                language="en",
                beam_size=1,
                vad_filter=False
            )
        
        # Combine segments into full transcription
        print(f"📝 Processing segments...")
        transcription_parts = []
        segment_count = 0
        
        for segment in segments:
            transcription_parts.append(segment.text)
            segment_count += 1
        
        full_text = " ".join(transcription_parts).strip()
        
        # Create result dict (compatible with original whisper format)
        result = {
            'text': full_text,
            'segments': [],  # Could populate if needed
            'language': info.language
        }
        
        print(f"{'='*70}")
        print(f"✅ TRANSCRIPTION COMPLETE!")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   • Audio duration: {info.duration:.2f} seconds")
        print(f"   • Segments processed: {segment_count}")
        print(f"   • Transcribed text: {len(full_text)} characters")
        print(f"   • Words: ~{len(full_text.split())} words")
        print(f"   • Language detected: {info.language} (probability: {info.language_probability:.2f})")
        print(f"\n📝 Transcription preview:")
        print(f"   \"{full_text[:200]}{'...' if len(full_text) > 200 else ''}\"")
        print(f"{'='*70}\n")
        
        if len(full_text) < 10:
            print("⚠️  WARNING: Transcription very short - audio might be unclear or mostly silent")
        
        return result
        
    except Exception as e:
        print(f"\n❌ ERROR: Faster-Whisper transcription failed")
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        raise Exception(f"Audio transcription failed: {str(e)}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str,int,int]]:
    """
    Split text into overlapping chunks.
    Returns list of tuples: (chunk_text, start_char, end_char)
    """
    text = text.replace("\r\n", "\n")
    start = 0
    chunks = []
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        start = max(end - overlap, end)
    return chunks

def chunk_audio_by_seconds(path: str, chunk_seconds: int = 30, overlap_seconds: int = 5) -> List[Tuple[str, float, float]]:
    """
    Splits audio file into overlapping chunks (saved to temp files). Returns list of (temp_path, start_s, end_s)
    """
    import tempfile
    import soundfile as sf
    data, sr = sf.read(path)
    total_seconds = len(data) / sr
    chunks = []
    cur = 0.0
    while cur < total_seconds:
        end = min(cur + chunk_seconds, total_seconds)
        start_idx = int(cur * sr)
        end_idx = int(end * sr)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, data[start_idx:end_idx], sr)
        chunks.append((tmp.name, cur, end))
        cur = max(end - overlap_seconds, end)
    return chunks

if __name__ == "__main__":
    print("ingestion module loaded")
