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
        
    def get_or_create_collection(self, name: str):
        # ChromaDB requires collection names to be alphanumeric, 3-63 chars.
        # Ensure we sanitize if necessary, though "embeddings_1800" is fine.
        return self.client.get_or_create_collection(name=name)
        
    def add_embeddings(self, collection_name: str, 
                       embeddings: List[np.ndarray], 
                       metadatas: List[Dict[str, Any]], 
                       ids: List[str]):
        """
        Adds a batch of embeddings to the specified collection.
        """
        if not embeddings:
            return
            
        collection = self.get_or_create_collection(collection_name)
        
        # ChromaDB expects lists, not numpy arrays
        embeddings_list = [e.tolist() if isinstance(e, np.ndarray) else e for e in embeddings]
        
        collection.add(
            embeddings=embeddings_list,
            metadatas=metadatas,
            ids=ids
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

    def delete_collection(self, collection_name: str):
        """Deletes a collection if it exists."""
        try:
            self.client.delete_collection(collection_name)
            print(f"Deleted collection: {collection_name}")
        except Exception as e:
            # Collection likely doesn't exist, which is fine
            print(f"Collection {collection_name} could not be deleted (might not exist): {e}")

    def list_collections(self) -> List[str]:
        """Returns a list of all collection names."""
        return [c.name for c in self.client.list_collections()]
