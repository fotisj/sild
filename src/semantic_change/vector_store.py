import chromadb
import chromadb.errors
import os
import time
import numpy as np
from typing import List, Dict, Any, Optional
import sys

class VectorStore:
    """
    Wrapper for ChromaDB to store and query word embeddings.
    """
    def __init__(self, persistence_path: str = "data/chroma_db"):
        self.persistence_path = persistence_path
        try:
            self.client = chromadb.PersistentClient(path=persistence_path)
        except Exception as e:
            print(f"Error initializing ChromaDB at '{persistence_path}':", file=sys.stderr)
            print(f"  ChromaDB version: {chromadb.__version__}", file=sys.stderr)
            print(f"  Error: {e}", file=sys.stderr)
            print("\nThis may be caused by a version mismatch between the ChromaDB that", file=sys.stderr)
            print("created the database and your local version.", file=sys.stderr)
            print("\nPossible solutions:", file=sys.stderr)
            print("  1. Ensure same ChromaDB version locally and on HPC", file=sys.stderr)
            print("  2. Delete data/chroma_db and regenerate embeddings", file=sys.stderr)
            print("  3. Pin chromadb version in requirements.txt", file=sys.stderr)
            raise
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
                       max_batch_size: int = 5000,
                       max_retries: int = 3):
        """
        Adds a batch of embeddings to the specified collection.
        Automatically splits into smaller batches if needed (ChromaDB limit ~5461).
        Includes retry logic for transient errors.
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
            
            # Retry loop
            for attempt in range(max_retries):
                try:
                    collection.add(
                        embeddings=embeddings_list[start:end],
                        metadatas=metadatas[start:end],
                        ids=ids[start:end]
                    )
                    break # Success, exit retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_time = 2 ** attempt # Exponential backoff: 1s, 2s, 4s
                        print(f"Warning: ChromaDB add failed (attempt {attempt+1}/{max_retries}). Retrying in {sleep_time}s... Error: {e}")
                        time.sleep(sleep_time)
                    else:
                        print(f"Error: Failed to add embeddings to ChromaDB after {max_retries} attempts.")
                        raise e
        
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

    def get_by_ids(self, collection_name: str, ids: List[str]):
        """
        Retrieves specific items by their IDs.
        """
        collection = self.get_or_create_collection(collection_name)
        return collection.get(
            ids=ids,
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
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception as e:
            print(f"Error listing ChromaDB collections: {e}", file=sys.stderr)
            print(f"  ChromaDB version: {chromadb.__version__}", file=sys.stderr)
            print(f"  Database path: {self.persistence_path}", file=sys.stderr)
            raise

    def get_available_models(self, project_id: str) -> List[str]:
        """
        Scans collections to find which models have been processed for a project.
        Returns a list of unique safe model names (e.g., 'bert_base_uncased').

        Args:
            project_id: 4-digit project identifier

        Collection naming: embeddings_{project_id}_t1_{model}, embeddings_{project_id}_t2_{model}
        """
        collections = self.list_collections()
        models = set()
        prefix = f"embeddings_{project_id}_"

        for c in collections:
            if not c.startswith(prefix):
                continue

            # Remove prefix: embeddings_{project_id}_
            rest = c[len(prefix):]
            parts = rest.split("_", 1)

            if len(parts) == 2:
                period, model_name = parts
                if period in ("t1", "t2") and model_name:
                    models.add(model_name)

        return sorted(list(models))

    def get_collection_names_for_model(
        self,
        project_id: str,
        model_name: str
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Get collection names for a project and model.

        Args:
            project_id: 4-digit project identifier
            model_name: Safe model name (e.g., 'bert_base_uncased')

        Returns:
            Tuple of (collection_t1, collection_t2) - either may be None if not found
        """
        collections = self.list_collections()

        coll_t1 = f"embeddings_{project_id}_t1_{model_name}"
        coll_t2 = f"embeddings_{project_id}_t2_{model_name}"

        found_t1 = coll_t1 if coll_t1 in collections else None
        found_t2 = coll_t2 if coll_t2 in collections else None

        return found_t1, found_t2
