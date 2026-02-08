"""
Business logic services for the Semantic Change Analysis toolkit.

This module provides service classes that encapsulate business logic
for statistics retrieval and cluster operations, keeping GUI code focused
on rendering.
"""
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Any
import numpy as np


@dataclass
class CorpusStats:
    """Statistics for a single corpus."""
    files: int
    sentences: int
    tokens: int


@dataclass
class EmbeddingStats:
    """Statistics for embeddings in ChromaDB."""
    model_name: str
    total_embeddings: int
    unique_lemmas: int
    count_t1: int
    count_t2: int


class StatsService:
    """Service for retrieving corpus and embedding statistics."""

    def __init__(self, chroma_path: str = "data/chroma_db"):
        """
        Initialize the stats service.

        Args:
            chroma_path: Path to the ChromaDB persistence directory
        """
        self.chroma_path = chroma_path

    def get_corpus_stats(self, db_path: str, label: str) -> Optional[CorpusStats]:
        """
        Get statistics for a single corpus.

        Args:
            db_path: Path to the SQLite corpus database
            label: Label for the corpus (e.g., "1800")

        Returns:
            CorpusStats object or None if unable to read
        """
        try:
            from semantic_change.corpus import Corpus
            corpus = Corpus(label, "", db_path)
            stats = corpus.get_stats()
            return CorpusStats(
                files=stats.get("files", 0),
                sentences=stats.get("sentences", 0),
                tokens=stats.get("tokens", 0)
            )
        except Exception:
            return None

    def get_embedding_stats(
        self,
        project_id: str,
        model_name: str
    ) -> Optional[EmbeddingStats]:
        """
        Get embedding statistics from ChromaDB for a specific model.

        Args:
            project_id: 4-digit project identifier
            model_name: HuggingFace model name or safe name

        Returns:
            EmbeddingStats object or None if unable to read
        """
        try:
            from semantic_change.vector_store import VectorStore
            store = VectorStore(persistence_path=self.chroma_path)

            # Sanitize model name for collection (matches how collections are created)
            safe_model = model_name.replace("/", "_").replace("-", "_")
            coll_t1 = f"embeddings_{project_id}_t1_{safe_model}"
            coll_t2 = f"embeddings_{project_id}_t2_{safe_model}"

            count_t1 = store.count(coll_t1)
            count_t2 = store.count(coll_t2)
            total_embeddings = count_t1 + count_t2

            # Count unique lemmas using batched queries to avoid memory issues
            unique_lemmas = self._count_unique_lemmas(store, coll_t1, coll_t2, count_t1, count_t2)

            return EmbeddingStats(
                model_name=model_name,
                total_embeddings=total_embeddings,
                unique_lemmas=unique_lemmas,
                count_t1=count_t1,
                count_t2=count_t2
            )
        except Exception:
            return None

    def _count_unique_lemmas(
        self,
        store: Any,
        coll_t1: str,
        coll_t2: str,
        count_t1: int,
        count_t2: int,
        batch_size: int = 10000
    ) -> int:
        """
        Count unique lemmas across two collections using batched queries.

        Args:
            store: VectorStore instance
            coll_t1: Name of collection for time period 1
            coll_t2: Name of collection for time period 2
            count_t1: Number of embeddings in collection 1
            count_t2: Number of embeddings in collection 2
            batch_size: Size of batches for querying

        Returns:
            Count of unique lemmas
        """
        c1 = store.get_or_create_collection(coll_t1)
        c2 = store.get_or_create_collection(coll_t2)

        all_lemmas = set()

        # Process each collection in batches
        for coll, count in [(c1, count_t1), (c2, count_t2)]:
            if count == 0:
                continue
            for offset in range(0, count, batch_size):
                batch = coll.get(include=["metadatas"], limit=batch_size, offset=offset)["metadatas"]
                for m in batch:
                    all_lemmas.add(m.get("lemma"))

        return len(all_lemmas)


class ClusterService:
    """Service for cluster-related operations."""

    @staticmethod
    def save_for_drilldown(
        embeddings: np.ndarray,
        sentences: np.ndarray,
        filenames: np.ndarray,
        time_labels: np.ndarray,
        sense_labels: np.ndarray,
        spans: Optional[List[Any]],
        metadata: dict,
        cluster_id: int,
        output_dir: str = "output"
    ) -> str:
        """
        Save a sense cluster's data for drill-down analysis.

        Args:
            embeddings: Combined embeddings array
            sentences: Combined sentences array
            filenames: Combined filenames array
            time_labels: Combined time labels array
            sense_labels: Array of cluster labels
            spans: Optional list of span data
            metadata: Metadata dict with project_id, model_name, target_word, etc.
            cluster_id: The cluster ID to save
            output_dir: Output directory for saved files

        Returns:
            Path to the saved .npz file
        """
        # Filter to selected cluster
        mask = sense_labels == cluster_id

        # Create output directory
        clusters_dir = os.path.join(output_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)

        # Generate filename
        model_short = metadata['model_name'].replace("/", "_").replace("-", "_")
        project_id = metadata.get('project_id', 'unknown')
        target_word = metadata.get('target_word', 'unknown')
        filename = f"k{project_id}_{model_short}_{target_word}_cluster{cluster_id}.npz"
        filepath = os.path.join(clusters_dir, filename)

        # Prepare spans data
        spans_data = None
        if spans is not None:
            spans_data = np.array([spans[i] for i, m in enumerate(mask) if m], dtype=object)

        # Save data
        np.savez(
            filepath,
            embeddings=embeddings[mask],
            sentences=sentences[mask],
            filenames=filenames[mask],
            time_labels=time_labels[mask],
            spans=spans_data,
            metadata=json.dumps({**metadata, 'original_cluster_id': int(cluster_id)})
        )

        return filepath
