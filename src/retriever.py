# src/retriever.py
import faiss
import json
from sentence_transformers import SentenceTransformer
from typing import List

from src.config import EMBEDDING_MODEL, DEVICE, FAISS_INDEX_PATH, CONTEXTS_PATH

class FaissRetriever:
    """
    A retriever class that uses FAISS for efficient similarity search.
    It loads a pre-built FAISS index and a list of contexts.
    """
    def __init__(self):
        print("Initializing FaissRetriever...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        print(f"  -> Loading FAISS index from: {FAISS_INDEX_PATH}")
        self.index = self._load_faiss_index()
        print(f"  -> Loading contexts from: {CONTEXTS_PATH}")
        self.contexts = self._load_contexts()
        print("FaissRetriever initialized successfully.")

    def _load_faiss_index(self):
        """Loads the FAISS index and moves it to the GPU if available."""
        index = faiss.read_index(FAISS_INDEX_PATH)
        if DEVICE == 'cuda':
            print("  -> Moving FAISS index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def _load_contexts(self) -> List[str]:
        """Loads the context documents."""
        with open(CONTEXTS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

    def search(self, query: str, top_k: int) -> List[str]:
        """
        Searches for the most similar contexts to a given query.
        
        Args:
            query (str): The user's question.
            top_k (int): The number of top results to return.
            
        Returns:
            List[str]: A list of the most relevant context strings.
        """
        print(f"Searching for top {top_k} results for query: '{query}'")
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        
        # Search the FAISS index
        _scores, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the actual context strings using the indices
        results = [self.contexts[i] for i in indices[0]]
        return results
