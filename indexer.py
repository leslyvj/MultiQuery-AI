# indexer.py
import os
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

DEFAULT_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

class ChromaIndexer:
    def __init__(self, persist_directory: str = DEFAULT_DIR):
        os.makedirs(persist_directory, exist_ok=True)
        # Use the new PersistentClient API
        self.client = chromadb.PersistentClient(path=persist_directory)
        # Primary collection for text/audio (384-dim sentence-transformers)
        self.collection = self.client.get_or_create_collection(name="multimodal_collection")
        # Secondary collection for CLIP visual embeddings (512-dim)
        self.clip_collection = self.client.get_or_create_collection(name="clip_visual_collection")

    def add_items(self, ids: List[str], embeddings: np.ndarray, metadatas: List[Dict[str, Any]], documents: List[str], use_clip: bool = False):
        """
        ids: list of unique ids (strings)
        embeddings: numpy array (N x dim)
        metadatas: list of dicts
        documents: list of text (can be transcripts or OCR or captions)
        use_clip: if True, use CLIP collection (512-dim), else use main collection (384-dim)
        """
        emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        
        # Choose collection based on embedding dimension
        collection = self.clip_collection if use_clip else self.collection
        
        collection.add(ids=ids, embeddings=emb_list, metadatas=metadatas, documents=documents)
        # PersistentClient automatically persists data

    def query(self, query_embeddings, n_results: int = 5, use_clip: bool = False):
        emb = query_embeddings.tolist() if hasattr(query_embeddings, "tolist") else query_embeddings
        
        # Choose collection based on query embedding dimension
        collection = self.clip_collection if use_clip else self.collection
        
        res = collection.query(query_embeddings=emb, n_results=n_results, include=['metadatas','documents','distances'])
        return res
    
    def query_both(self, query_emb_text, query_emb_clip, n_results: int = 5):
        """
        Query both collections and merge results.
        Used for cross-format retrieval across different embedding spaces.
        """
        # Query text collection (384-dim)
        res_text = self.collection.query(
            query_embeddings=query_emb_text.tolist() if hasattr(query_emb_text, "tolist") else query_emb_text,
            n_results=n_results,
            include=['metadatas','documents','distances']
        )
        
        # Query CLIP collection (512-dim)
        res_clip = self.clip_collection.query(
            query_embeddings=query_emb_clip.tolist() if hasattr(query_emb_clip, "tolist") else query_emb_clip,
            n_results=n_results,
            include=['metadatas','documents','distances']
        )
        
        # Merge results
        merged = {
            'ids': [res_text.get('ids', [[]])[0] + res_clip.get('ids', [[]])[0]],
            'documents': [res_text.get('documents', [[]])[0] + res_clip.get('documents', [[]])[0]],
            'metadatas': [res_text.get('metadatas', [[]])[0] + res_clip.get('metadatas', [[]])[0]],
            'distances': [res_text.get('distances', [[]])[0] + res_clip.get('distances', [[]])[0]]
        }
        
        return merged

    def delete_collection(self):
        self.client.delete_collection("multimodal_collection")

if __name__ == "__main__":
    print("Chroma indexer ready")
