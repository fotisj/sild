from typing import Optional, List
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from .vector_store import VectorStore

def compute_centroid(embeddings) -> Optional[np.ndarray]:
    """
    Computes the centroid (mean vector) of a list of embeddings.

    Args:
        embeddings: List of embeddings (can be ChromaDB PyEmbedding or numpy arrays).

    Returns:
        The centroid as a numpy array, or None if embeddings list is empty.
    """
    if embeddings is None or len(embeddings) == 0:
        return None
    embeddings_array = np.array(embeddings)
    return np.mean(embeddings_array, axis=0)


def compute_semantic_change_centroid_distance(
    collection_name_t1: str,
    collection_name_t2: str,
    lemma: str,
    vector_store: VectorStore,
    pos: Optional[str] = None,
) -> Optional[float]:
    """
    Computes semantic change between two periods using centroid distance.

    Retrieves all embeddings for a lemma from both time periods, computes
    the centroid for each period, and returns the cosine distance between them.

    Args:
        collection_name_t1: ChromaDB collection name for period 1.
        collection_name_t2: ChromaDB collection name for period 2.
        lemma: The word lemma to analyze.
        vector_store: VectorStore instance for querying embeddings.
        pos: Optional POS filter (Currently unused due to metadata limitations).

    Returns:
        The cosine distance between the two period centroids, or None if
        embeddings could not be retrieved for one or both periods.
    """
    # Query embeddings for period 1 (case-sensitive to support cased models)
    if pos:
        where_clause_t1 = {"$and": [{"lemma": lemma}, {"pos": pos}]}
    else:
        where_clause_t1 = {"lemma": lemma}

    results_t1 = vector_store.get_by_metadata(
        collection_name_t1, where=where_clause_t1, limit=100000
    )
    embeddings_t1 = results_t1.get("embeddings", [])

    # Query embeddings for period 2 (case-sensitive to support cased models)
    if pos:
        where_clause_t2 = {"$and": [{"lemma": lemma}, {"pos": pos}]}
    else:
        where_clause_t2 = {"lemma": lemma}

    results_t2 = vector_store.get_by_metadata(
        collection_name_t2, where=where_clause_t2, limit=100000
    )
    embeddings_t2 = results_t2.get("embeddings", [])

    # Compute centroids
    centroid_t1 = compute_centroid(embeddings_t1)
    centroid_t2 = compute_centroid(embeddings_t2)

    if centroid_t1 is None or centroid_t2 is None:
        return None

    # Compute cosine distance
    distance = cosine_distances([centroid_t1], [centroid_t2])[0][0]
    return float(distance)


def compute_semantic_change(
    collection_name_t1: str,
    collection_name_t2: str,
    lemma: str,
    vector_store: VectorStore,
    algorithm: str = "centroid_distance",
    pos: Optional[str] = None,
) -> Optional[float]:
    """
    Computes semantic change between two periods using the specified algorithm.

    Args:
        collection_name_t1: ChromaDB collection name for period 1.
        collection_name_t2: ChromaDB collection name for period 2.
        lemma: The word lemma to analyze.
        vector_store: VectorStore instance for querying embeddings.
        algorithm: The algorithm to use for semantic change computation.
                   Options: "centroid_distance" (default).
        pos: Optional POS filter.

    Returns:
        The semantic change score, or None if computation failed.
    """
    if algorithm == "centroid_distance":
        return compute_semantic_change_centroid_distance(
            collection_name_t1, collection_name_t2, lemma, vector_store, pos
        )
    else:
        raise ValueError(f"Unknown semantic change algorithm: {algorithm}")
