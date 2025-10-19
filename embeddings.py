# embeddings.py
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import List, Union

TEXT_MODEL = "all-MiniLM-L6-v2"
IMAGE_MODEL = "clip-ViT-B-32"

class Embedder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.text_model = SentenceTransformer(TEXT_MODEL, device=device)
        self.image_model = SentenceTransformer(IMAGE_MODEL, device=device)

    def embed_text(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            texts = [texts]
        emb = self.text_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return emb

    def embed_image(self, pil_image: Image.Image):
        # sentence-transformers CLIP models accept PIL images in encode
        emb = self.image_model.encode(pil_image, convert_to_numpy=True, show_progress_bar=False)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        return emb

if __name__ == "__main__":
    e = Embedder(device="cpu")
    print("Models loaded")
