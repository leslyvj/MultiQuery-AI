"""
Advanced OCR Engine for Multimodal RAG
Supports multiple OCR backends with automatic fallback:
1. EasyOCR (primary) - Deep learning, GPU accelerated, 80+ languages
2. PaddleOCR (secondary) - Fast and accurate
3. Tesseract (fallback) - Traditional OCR
"""

import os
import warnings
from typing import Optional, List, Tuple
from PIL import Image
import numpy as np

warnings.filterwarnings('ignore')


class OCREngine:
    """Advanced OCR with multiple backend support"""
    
    def __init__(self):
        self.backend = None
        self.reader = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Try to initialize OCR backends in order of preference"""
        
        # Try EasyOCR first (best for complex text, GPU accelerated)
        try:
            import easyocr
            print("🔍 Initializing EasyOCR (GPU accelerated)...")
            # Initialize with English, you can add more languages: ['en', 'hi', 'ar', ...]
            self.reader = easyocr.Reader(['en'], gpu=True, verbose=False)
            self.backend = 'easyocr'
            print("✅ EasyOCR initialized successfully!")
            return
        except ImportError:
            print("⚠️  EasyOCR not installed. Try: pip install easyocr")
        except Exception as e:
            print(f"⚠️  EasyOCR initialization failed: {e}")
        
        # Try PaddleOCR as secondary option
        try:
            from paddleocr import PaddleOCR
            print("🔍 Initializing PaddleOCR...")
            self.reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            self.backend = 'paddleocr'
            print("✅ PaddleOCR initialized successfully!")
            return
        except ImportError:
            print("⚠️  PaddleOCR not installed. Try: pip install paddleocr")
        except Exception as e:
            print(f"⚠️  PaddleOCR initialization failed: {e}")
        
        # Fallback to Tesseract
        try:
            import pytesseract
            # Test if tesseract is actually available
            pytesseract.get_tesseract_version()
            self.reader = pytesseract
            self.backend = 'tesseract'
            print("✅ Tesseract OCR initialized successfully!")
            return
        except Exception as e:
            print(f"⚠️  Tesseract not available: {e}")
        
        print("❌ No OCR backend available. Images will be indexed with visual embeddings only.")
        self.backend = None
    
    def extract_text(self, image_path: str) -> str:
        """
        Extract text from image using available OCR backend
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text string
        """
        if self.backend is None:
            return ""
        
        try:
            # Load image
            if isinstance(image_path, str):
                img = Image.open(image_path)
            else:
                img = image_path
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Extract text based on backend
            if self.backend == 'easyocr':
                return self._extract_with_easyocr(img)
            elif self.backend == 'paddleocr':
                return self._extract_with_paddleocr(img)
            elif self.backend == 'tesseract':
                return self._extract_with_tesseract(img)
            
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            return ""
        
        return ""
    
    def _extract_with_easyocr(self, img: Image.Image) -> str:
        """Extract text using EasyOCR"""
        try:
            # Convert PIL to numpy array
            img_array = np.array(img)
            
            # Perform OCR
            results = self.reader.readtext(img_array, detail=0, paragraph=True)
            
            # Combine all text
            text = ' '.join(results)
            
            print(f"  📝 EasyOCR extracted {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            print(f"  ❌ EasyOCR error: {e}")
            return ""
    
    def _extract_with_paddleocr(self, img: Image.Image) -> str:
        """Extract text using PaddleOCR"""
        try:
            # Convert PIL to numpy array
            img_array = np.array(img)
            
            # Perform OCR
            results = self.reader.ocr(img_array, cls=True)
            
            # Extract text from results
            text_parts = []
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        text_parts.append(line[1][0])  # line[1][0] is the text
            
            text = ' '.join(text_parts)
            
            print(f"  📝 PaddleOCR extracted {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            print(f"  ❌ PaddleOCR error: {e}")
            return ""
    
    def _extract_with_tesseract(self, img: Image.Image) -> str:
        """Extract text using Tesseract"""
        try:
            import pytesseract
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            
            print(f"  📝 Tesseract extracted {len(text)} characters")
            return text.strip()
            
        except Exception as e:
            print(f"  ❌ Tesseract error: {e}")
            return ""
    
    def extract_with_confidence(self, image_path: str) -> List[Tuple[str, float]]:
        """
        Extract text with confidence scores (only for EasyOCR)
        
        Returns:
            List of (text, confidence) tuples
        """
        if self.backend != 'easyocr':
            text = self.extract_text(image_path)
            return [(text, 1.0)] if text else []
        
        try:
            img = Image.open(image_path) if isinstance(image_path, str) else image_path
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_array = np.array(img)
            results = self.reader.readtext(img_array, detail=1)
            
            # Results format: [([coordinates], text, confidence), ...]
            text_confidence = [(text, conf) for (_, text, conf) in results]
            
            return text_confidence
            
        except Exception as e:
            print(f"❌ OCR with confidence failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if any OCR backend is available"""
        return self.backend is not None
    
    def get_backend(self) -> Optional[str]:
        """Get current OCR backend name"""
        return self.backend


# Global OCR engine instance
_ocr_engine = None


def get_ocr_engine() -> OCREngine:
    """Get or create global OCR engine instance"""
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = OCREngine()
    return _ocr_engine


# Test function
def test_ocr():
    """Test OCR engine"""
    engine = get_ocr_engine()
    
    print("\n" + "="*70)
    print("🧪 OCR ENGINE TEST")
    print("="*70)
    print(f"Backend: {engine.get_backend()}")
    print(f"Available: {engine.is_available()}")
    print("="*70)
    
    if engine.is_available():
        print("\n✅ OCR is ready to extract text from images!")
        print(f"   Using: {engine.backend}")
    else:
        print("\n❌ No OCR backend available")
        print("\nInstall one of these:")
        print("  pip install easyocr           # Recommended (GPU accelerated)")
        print("  pip install paddleocr          # Alternative (fast)")
        print("  pip install pytesseract        # Traditional")


if __name__ == "__main__":
    test_ocr()
