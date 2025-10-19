"""Run the server - keeps it alive"""
import uvicorn
import os

# Add FFmpeg to PATH
if os.path.exists("C:\\ffmpeg\\bin"):
    os.environ["PATH"] = "C:\\ffmpeg\\bin;" + os.environ.get("PATH", "")

print("="*60)
print("Starting Multimodal RAG Server on http://127.0.0.1:8000")
print("="*60)
print("\nServer is running. Press CTRL+C to stop.")
print("\nOpen your browser to: http://127.0.0.1:8000")
print("\nFixed issues:")
print("  - FFmpeg added to PATH for audio processing")
print("  - Tesseract OCR errors handled gracefully")
print("  - Images will be indexed without OCR text if Tesseract not installed")
print("="*60)
print()

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, log_level="info")
