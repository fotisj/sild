import chromadb
import os
import numpy as np
from typing import List, Dict, Any, Optional

class VectorStore:
    """
    Wrapper for ChromaDB to store and query word embeddings.
    """
    def __init__(self, persistence_path: str = "data/chroma_db"):
        self.client = chromadb.PersistentClient(path=persistence_path)
        self._collection_cache: Dict[str, Any] = {}

    def get_or_create_collection(self, name: str):
        # ChromaDB requires collection names to be alphanumeric, 3-63 chars.
        # Cache collection objects to avoid repeated lookups
        if name not in self._collection_cache:
            self._collection_cache[name] = self.client.get_or_create_collection(name=name)
        return self._collection_cache[name]
        
    def add_embeddings(self, collection_name: str,
                       embeddings: List[np.ndarray],
                       metadatas: List[Dict[str, Any]],
                       ids: List[str],
                       max_batch_size: int = 5000):
        """
        Adds a batch of embeddings to the specified collection.
        Automatically splits into smaller batches if needed (ChromaDB limit ~5461).
        """
        if embeddings is None or len(embeddings) == 0:
            return

        collection = self.get_or_create_collection(collection_name)

        # ChromaDB expects lists - use vectorized conversion via numpy stack
        if len(embeddings) > 0 and isinstance(embeddings[0], np.ndarray):
            # Stack into single array then convert - much faster than per-item conversion
            embeddings_list = np.stack(embeddings).tolist()
        else:
            embeddings_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]

        # Split into batches if exceeding ChromaDB's max batch size
        total = len(embeddings_list)
        for start in range(0, total, max_batch_size):
            end = min(start + max_batch_size, total)
            collection.add(
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end]
            )
        
    def query(self, collection_name: str, query_embeddings: List[np.ndarray], n_results: int = 10, where: Dict = None):
        """
        Queries the collection for nearest neighbors.
        """
        collection = self.get_or_create_collection(collection_name)
        
        embeddings_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in query_embeddings]
        
        return collection.query(
            query_embeddings=embeddings_list,
            n_results=n_results,
            where=where
        )

    def get_by_metadata(self, collection_name: str, where: Dict, limit: int = 100):
        """
        Retrieves items based on metadata filtering.
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.get(
            where=where,
            limit=limit,
            include=["embeddings", "metadatas"]
        )

    def count(self, collection_name: str) -> int:
        return self.get_or_create_collection(collection_name).count()

    def delete_collection(self, collection_name: str) -> tuple[bool, str]:
        """
        Deletes a collection if it exists.

        Returns:
            Tuple of (success, message)
        """
        try:
            self.client.delete_collection(collection_name)
            # Invalidate cache
            self._collection_cache.pop(collection_name, None)
            return True, f"Deleted: {collection_name}"
        except Exception as e:
            # Collection likely doesn't exist, which is fine
            self._collection_cache.pop(collection_name, None)
            return False, f"{collection_name} not found"

    def list_collections(self) -> List[str]:
        """Returns a list of all collection names."""
        return [c.name for c in self.client.list_collections()]

    def get_available_models(self) -> List[str]:
        """
        Scans collections to find which models have been processed.
        Returns a list of unique safe model names (e.g., 'bert_base_uncased').
        """
        collections = self.list_collections()
        models = set()
        for c in collections:
            if c.startswith("embeddings_t1_"):
                safe_name = c.replace("embeddings_t1_", "")
                models.add(safe_name)
        return sorted(list(models))
