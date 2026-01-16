"""
Semantic Change Analysis - Main Module

This module provides functions for analyzing semantic change of words
across time periods using contextual embeddings and clustering.
"""

import os
import glob
import numpy as np
import argparse
import sqlite3
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

from semantic_change.corpus import CorpusManager
from semantic_change.wsi import WordSenseInductor
from semantic_change.visualization import Visualizer
from semantic_change.vector_store import VectorStore


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for semantic change analysis."""
    project_id: str = ""  # 4-digit project identifier (required)
    target_word: str = "current"
    db_path_t1: str = "data/corpus_t1.db"
    db_path_t2: str = "data/corpus_t2.db"
    period_t1_label: str = "1800"
    period_t2_label: str = "1900"
    model_name: str = "bert-base-uncased"
    n_samples: int = 50
    k_neighbors: int = 10
    min_cluster_size: int = 3
    n_clusters: int = 3
    wsi_algorithm: str = "hdbscan"
    pos_filter: Optional[str] = None
    clustering_reduction: Optional[str] = None
    clustering_n_components: int = 50
    viz_reduction: str = "pca"
    viz_max_instances: int = 100
    context_window: int = 0
    output_dir: str = "output"


@dataclass
class EmbeddingData:
    """Container for embeddings and their associated metadata."""
    embeddings: np.ndarray
    sentences: List[str]
    spans: Optional[List[Optional[Tuple[int, int]]]]
    filenames: List[str]
    period_label: str

    def __len__(self) -> int:
        return len(self.embeddings)

    def is_empty(self) -> bool:
        return len(self.embeddings) == 0


@dataclass
class CombinedEmbeddingData:
    """Combined embedding data from multiple time periods."""
    embeddings: np.ndarray
    sentences: np.ndarray
    filenames: np.ndarray
    time_labels: np.ndarray
    highlight_spans: Optional[List[Optional[Tuple[int, int]]]]


@dataclass
class VisualizationData:
    """Data prepared for visualization after subsampling and reduction."""
    embeddings_2d: np.ndarray
    sense_labels: np.ndarray
    time_labels: np.ndarray
    sentences: np.ndarray
    filenames: np.ndarray
    highlight_spans: Optional[List[Optional[Tuple[int, int]]]]
    original_indices: np.ndarray


# =============================================================================
# Utility Functions
# =============================================================================

def clean_output_directory(output_dir: str) -> None:
    """Remove old visualization files from the output directory."""
    for f in glob.glob(os.path.join(output_dir, "neighbors_cluster_*.html")):
        try:
            os.remove(f)
        except OSError:
            pass

    for filename in ["sense_clusters.html", "time_period.html", "sense_time_combined.html"]:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError:
                pass


def get_collection_name(project_id: str, period: str, model_name: str) -> str:
    """
    Generate a standardized collection name for the vector store.

    Args:
        project_id: 4-digit project identifier
        period: "t1" or "t2"
        model_name: HuggingFace model name

    Returns:
        Collection name in format: embeddings_{project_id}_{period}_{safe_model}
    """
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_{project_id}_{period}_{safe_model}"


# =============================================================================
# Embedding Retrieval Functions
# =============================================================================

def fetch_sentences_from_db(
    db_path: str,
    sentence_ids: List[int]
) -> Tuple[List[str], List[str]]:
    """
    Fetch sentence texts and filenames from SQLite database.

    Args:
        db_path: Path to the SQLite database
        sentence_ids: List of sentence IDs to fetch

    Returns:
        Tuple of (sentences, filenames)
    """
    sentences = []
    filenames = []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for sent_id in sentence_ids:
        try:
            row = cursor.execute("""
                SELECT s.text, f.filename
                FROM sentences s
                JOIN files f ON s.file_id = f.id
                WHERE s.id=?
            """, (sent_id,)).fetchone()

            if row:
                sentences.append(row[0])
                filenames.append(row[1])
            else:
                sentences.append("[Missing Sentence]")
                filenames.append("Unknown")
        except sqlite3.OperationalError:
            row = cursor.execute(
                "SELECT text FROM sentences WHERE id=?", (sent_id,)
            ).fetchone()
            sentences.append(row[0] if row else "[Missing Sentence]")
            filenames.append("Unknown")

    conn.close()
    return sentences, filenames


def filter_embeddings_by_pos(
    embeddings: List,
    metadatas: List[Dict],
    pos_filter: Optional[str],
    n_samples: int
) -> Tuple[List[int], np.ndarray, List[Dict]]:
    """
    Filter embedding data by POS tag.

    Args:
        embeddings: List of embedding vectors
        metadatas: List of metadata dicts
        pos_filter: POS tag to filter by (e.g., 'NOUN', 'VERB') or None
        n_samples: Maximum number of samples to return

    Returns:
        Tuple of (indices_kept, filtered_embeddings, filtered_metadatas)
    """
    use_strict_query = pos_filter and pos_filter != 'NOUN'
    indices_to_keep = []

    for i, m in enumerate(metadatas):
        m_pos = m.get('pos')
        if use_strict_query:
            indices_to_keep.append(i)
        elif pos_filter == 'NOUN':
            if m_pos is None or m_pos == 'NOUN':
                indices_to_keep.append(i)
        else:
            indices_to_keep.append(i)

    indices_to_keep = indices_to_keep[:n_samples]

    filtered_embeddings = np.array([embeddings[i] for i in indices_to_keep])
    filtered_metadatas = [metadatas[i] for i in indices_to_keep]

    return indices_to_keep, filtered_embeddings, filtered_metadatas


def extract_spans_from_metadata(
    metadatas: List[Dict]
) -> List[Optional[Tuple[int, int]]]:
    """Extract character span tuples from metadata list."""
    spans = []
    for m in metadatas:
        start_char = m.get('start_char')
        end_char = m.get('end_char')
        if start_char is not None and end_char is not None:
            spans.append((start_char, end_char))
        else:
            spans.append(None)
    return spans


def fetch_embeddings_from_store(
    vector_store: VectorStore,
    collection_name: str,
    db_path: str,
    target_word: str,
    n_samples: int,
    pos_filter: Optional[str],
    context_window: int,
    period_label: str
) -> Optional[EmbeddingData]:
    """
    Fetch embeddings from vector store for a target word.

    Args:
        vector_store: VectorStore instance
        collection_name: Name of the collection to query
        db_path: Path to SQLite database for sentence retrieval
        target_word: Word to look up
        n_samples: Maximum samples to retrieve
        pos_filter: Optional POS filter
        context_window: Context window size (0 = sentence only)
        period_label: Label for this time period

    Returns:
        EmbeddingData if found, None otherwise
    """
    if context_window > 0:
        return None

    word_lower = target_word.lower()
    use_strict_query = pos_filter and pos_filter != 'NOUN'

    if use_strict_query:
        where_clause = {"$and": [{"lemma": word_lower}, {"pos": pos_filter}]}
    else:
        where_clause = {"lemma": word_lower}

    limit_request = n_samples * 2 if not use_strict_query else n_samples

    try:
        data = vector_store.get_by_metadata(
            collection_name, where=where_clause, limit=limit_request
        )
    except Exception as e:
        print(f"Warning: Vector store query failed: {e}")
        return None

    # ChromaDB returns a dict-like object where 'embeddings' may be None
    if not data:
        return None
    embeddings_raw = data.get('embeddings') if hasattr(data, 'get') else data['embeddings']
    metadatas_raw = data.get('metadatas') if hasattr(data, 'get') else data['metadatas']
    if embeddings_raw is None or len(embeddings_raw) == 0:
        return None

    indices, embeddings, metadatas = filter_embeddings_by_pos(
        embeddings_raw, metadatas_raw, pos_filter, n_samples
    )

    if len(embeddings) == 0:
        print(f"Found embeddings for '{target_word}' but none matched POS='{pos_filter}'")
        return None

    print(f"Found {len(embeddings)} cached embeddings in {collection_name} (after filtering)")

    sentence_ids = [m['sentence_id'] for m in metadatas]
    sentences, filenames = fetch_sentences_from_db(db_path, sentence_ids)
    spans = extract_spans_from_metadata(metadatas)

    return EmbeddingData(
        embeddings=embeddings,
        sentences=sentences,
        spans=spans,
        filenames=filenames,
        period_label=period_label
    )


# =============================================================================
# Embedding Combination Functions
# =============================================================================

def combine_embedding_data(
    data_t1: EmbeddingData,
    data_t2: EmbeddingData
) -> CombinedEmbeddingData:
    """
    Combine embedding data from two time periods.

    Args:
        data_t1: Embedding data from first time period
        data_t2: Embedding data from second time period

    Returns:
        CombinedEmbeddingData with all embeddings merged
    """
    embeddings = np.vstack([data_t1.embeddings, data_t2.embeddings])

    time_labels = np.array(
        [data_t1.period_label] * len(data_t1) +
        [data_t2.period_label] * len(data_t2)
    )

    sentences = np.array(data_t1.sentences + data_t2.sentences)
    filenames = np.array(data_t1.filenames + data_t2.filenames)

    if data_t1.spans is not None and data_t2.spans is not None:
        highlight_spans = data_t1.spans + data_t2.spans
    elif data_t1.spans is not None:
        highlight_spans = data_t1.spans + [None] * len(data_t2)
    elif data_t2.spans is not None:
        highlight_spans = [None] * len(data_t1) + data_t2.spans
    else:
        highlight_spans = None

    return CombinedEmbeddingData(
        embeddings=embeddings,
        sentences=sentences,
        filenames=filenames,
        time_labels=time_labels,
        highlight_spans=highlight_spans
    )


# =============================================================================
# Dimensionality Reduction Functions
# =============================================================================

def apply_dimensionality_reduction(
    embeddings: np.ndarray,
    method: Optional[str],
    n_components: int,
    random_state: int = 42
) -> np.ndarray:
    """
    Apply dimensionality reduction to embeddings.

    Args:
        embeddings: Input embeddings array
        method: Reduction method ('pca', 'umap', 'tsne') or None
        n_components: Target number of dimensions
        random_state: Random seed for reproducibility

    Returns:
        Reduced embeddings, or original if no reduction applied
    """
    if method is None:
        return embeddings

    if len(embeddings) <= n_components:
        print(f"Warning: Not enough data ({len(embeddings)}) for {n_components} "
              f"components, skipping reduction.")
        return embeddings

    method_lower = method.lower()

    if method_lower == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(embeddings)

    elif method_lower == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            return reducer.fit_transform(embeddings)
        except ImportError:
            print("Warning: umap-learn not installed. Skipping reduction.")
            return embeddings

    elif method_lower == 'tsne':
        from sklearn.manifold import TSNE
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(
            n_components=min(n_components, 3),
            perplexity=perplexity,
            random_state=random_state
        )
        return reducer.fit_transform(embeddings)

    else:
        print(f"Warning: Unknown reduction method '{method}', skipping.")
        return embeddings


# =============================================================================
# Clustering Functions
# =============================================================================

def run_word_sense_induction(
    embeddings: np.ndarray,
    algorithm: str,
    min_cluster_size: int,
    n_clusters: int
) -> np.ndarray:
    """
    Run Word Sense Induction clustering on embeddings.

    Args:
        embeddings: Input embeddings
        algorithm: Clustering algorithm name
        min_cluster_size: Minimum cluster size (for HDBSCAN)
        n_clusters: Number of clusters (for KMeans etc.)

    Returns:
        Array of cluster labels
    """
    wsi = WordSenseInductor(
        algorithm=algorithm,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters
    )
    return wsi.fit_predict(embeddings)


# =============================================================================
# Visualization Functions
# =============================================================================

def subsample_for_visualization(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sentences: np.ndarray,
    filenames: np.ndarray,
    max_per_class: int,
    spans: Optional[List] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[List], np.ndarray]:
    """
    Subsample data for visualization, ensuring balanced representation per class.

    Args:
        embeddings: Full embeddings array
        labels: Labels to balance by
        sentences: Sentence texts
        filenames: Source filenames
        max_per_class: Maximum samples per unique label
        spans: Optional highlight spans
        random_state: Random seed

    Returns:
        Tuple of (embeddings, labels, sentences, filenames, spans, indices)
    """
    np.random.seed(random_state)
    unique_labels = np.unique(labels)
    indices_to_keep = []

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        if len(label_indices) > max_per_class:
            selected = np.random.choice(label_indices, max_per_class, replace=False)
            indices_to_keep.extend(selected)
        else:
            indices_to_keep.extend(label_indices)

    indices = np.array(indices_to_keep)

    sub_spans = None
    if spans is not None:
        sub_spans = [spans[i] for i in indices_to_keep]

    return (
        embeddings[indices],
        labels[indices],
        sentences[indices],
        filenames[indices],
        sub_spans,
        indices
    )


def create_visualizations(
    viz_data: VisualizationData,
    target_word: str,
    output_dir: str,
    viz_method: str
) -> None:
    """
    Create all visualization plots.

    Args:
        viz_data: Prepared visualization data
        target_word: Word being analyzed
        output_dir: Directory to save plots
        viz_method: Visualization dimensionality reduction method
    """
    viz = Visualizer(method=viz_method)

    print("Plotting Clustering by Time...")
    viz.plot_clustering(
        viz_data.embeddings_2d,
        labels=viz_data.time_labels,
        sentences=viz_data.sentences,
        filenames=viz_data.filenames,
        title=f"'{target_word}' by Time Period",
        save_path=os.path.join(output_dir, "time_period.html"),
        highlight_spans=viz_data.highlight_spans
    )

    print("Plotting Clustering by Sense...")
    viz.plot_clustering(
        viz_data.embeddings_2d,
        labels=viz_data.sense_labels,
        sentences=viz_data.sentences,
        filenames=viz_data.filenames,
        title=f"'{target_word}' by Sense Cluster",
        save_path=os.path.join(output_dir, "sense_clusters.html"),
        highlight_spans=viz_data.highlight_spans
    )

    print("Plotting Combined Sense × Time...")
    viz.plot_combined_clustering(
        viz_data.embeddings_2d,
        sense_labels=viz_data.sense_labels,
        time_labels=list(viz_data.time_labels),
        sentences=viz_data.sentences,
        filenames=viz_data.filenames,
        title=f"'{target_word}' by Sense × Time",
        save_path=os.path.join(output_dir, "sense_time_combined.html"),
        highlight_spans=viz_data.highlight_spans
    )


# =============================================================================
# Neighbor Analysis Functions
# =============================================================================

def get_chroma_neighbors(
    vector_store: VectorStore,
    collection_name: str,
    centroid: np.ndarray,
    target_word: Optional[str] = None,
    n_neighbors: int = 10,
    max_candidates: int = 500
) -> Dict[str, np.ndarray]:
    """
    Find nearest neighbors by querying the vector store.

    Args:
        vector_store: VectorStore instance
        collection_name: Collection to query
        centroid: Query vector (cluster centroid)
        target_word: Word to exclude from results
        n_neighbors: Number of neighbors to return
        max_candidates: Maximum candidates to consider

    Returns:
        Dict mapping lemma to mean embedding vector
    """
    skip_lemmas = set()
    if target_word:
        skip_lemmas.add(target_word.lower())

    try:
        results = vector_store.query(
            collection_name=collection_name,
            query_embeddings=[centroid],
            n_results=max_candidates
        )
    except Exception as e:
        print(f"Vector store query failed: {e}")
        return {}

    if not results or not results['metadatas']:
        return {}

    metas = results['metadatas'][0]
    vecs = results['embeddings'][0] if results.get('embeddings') else []

    lemma_counts = Counter()
    lemma_vectors_sum = defaultdict(lambda: np.zeros(centroid.shape))

    for i, m in enumerate(metas):
        lemma = m.get('lemma', '').lower()
        if not lemma or len(lemma) < 2:
            continue
        if lemma in skip_lemmas:
            continue

        lemma_counts[lemma] += 1

        if vecs:
            v = np.array(vecs[i])
            lemma_vectors_sum[lemma] += v

    top_unique = []
    seen = set()

    for m in metas:
        lemma = m.get('lemma', '').lower()
        if not lemma or len(lemma) < 2:
            continue
        if lemma in skip_lemmas:
            continue
        if lemma in seen:
            continue

        seen.add(lemma)
        mean_vec = lemma_vectors_sum[lemma] / lemma_counts[lemma]
        top_unique.append((lemma, mean_vec))

        if len(top_unique) >= n_neighbors:
            break

    return {lemma: vec for lemma, vec in top_unique}


def create_neighbor_plots(
    embeddings: np.ndarray,
    sense_labels: np.ndarray,
    target_word: str,
    vector_store: Optional[VectorStore],
    collection_t1: str,
    collection_t2: str,
    k_neighbors: int,
    output_dir: str
) -> None:
    """
    Create neighbor visualization plots for each sense cluster.

    Args:
        embeddings: All embeddings
        sense_labels: Cluster labels
        target_word: Target word being analyzed
        vector_store: VectorStore instance (or None)
        collection_t1: Collection name for first period
        collection_t2: Collection name for second period
        k_neighbors: Number of neighbors per cluster
        output_dir: Output directory for plots
    """
    viz = Visualizer()
    unique_clusters = sorted(set(sense_labels))

    for cluster_id in unique_clusters:
        mask = sense_labels == cluster_id
        cluster_embs = embeddings[mask]

        if len(cluster_embs) == 0:
            continue

        centroid = np.mean(cluster_embs, axis=0)
        neighbors = {}
        source_label = "ChromaDB"

        if vector_store:
            try:
                print(f"Querying ChromaDB for cluster {cluster_id} neighbors...")
                n1 = get_chroma_neighbors(
                    vector_store, collection_t1, centroid,
                    target_word=target_word, n_neighbors=k_neighbors
                )
                n2 = get_chroma_neighbors(
                    vector_store, collection_t2, centroid,
                    target_word=target_word, n_neighbors=k_neighbors
                )

                merged = {**n1, **n2}
                neighbors = dict(list(merged.items())[:k_neighbors * 2])
            except Exception as e:
                print(f"ChromaDB lookup failed: {e}")

        print(f"Cluster {cluster_id} Neighbors ({source_label}): {list(neighbors.keys())}")

        viz.plot_neighbors(
            centroid,
            neighbors,
            centroid_name=target_word,
            title=f"Semantic Neighbors for Cluster {cluster_id} ({source_label})",
            save_path=os.path.join(output_dir, f"neighbors_cluster_{cluster_id}.html")
        )


# =============================================================================
# Main Analysis Functions
# =============================================================================

def validate_config(config: AnalysisConfig) -> Optional[str]:
    """
    Validate analysis configuration.

    Args:
        config: Configuration to validate

    Returns:
        Error message if invalid, None if valid
    """
    if not os.path.exists(config.db_path_t1):
        return f"Database not found: {config.db_path_t1}"
    if not os.path.exists(config.db_path_t2):
        return f"Database not found: {config.db_path_t2}"
    return None


def run_single_analysis(
    project_id: str,
    target_word: str = "current",
    db_path_t1: str = "data/corpus_t1.db",
    db_path_t2: str = "data/corpus_t2.db",
    period_t1_label: str = "1800",
    period_t2_label: str = "1900",
    model_name: str = "bert-base-uncased",
    k_neighbors: int = 10,
    min_cluster_size: int = 3,
    n_clusters: int = 3,
    wsi_algorithm: str = "hdbscan",
    pos_filter: Optional[str] = None,
    clustering_reduction: Optional[str] = None,
    clustering_n_components: int = 50,
    viz_reduction: str = "pca",
    n_samples: int = 50,
    viz_max_instances: int = 100,
    context_window: int = 0,
    # Legacy parameters (deprecated)
    use_umap: Optional[bool] = None,
    umap_n_components: Optional[int] = None,
    db_path_1800: Optional[str] = None,
    db_path_1900: Optional[str] = None,
    embedder=None,
    n_top_sentences: int = 10,
    k_per_sentence: int = 6
) -> None:
    """
    Run semantic change analysis for a single word.

    This is the main entry point that orchestrates the full analysis pipeline:
    1. Clean output directory
    2. Fetch embeddings from vector store
    3. Combine embeddings from both time periods
    4. Run word sense induction clustering
    5. Create visualizations
    6. Generate neighbor plots
    """
    # Handle legacy parameters
    if db_path_1800 is not None:
        db_path_t1 = db_path_1800
    if db_path_1900 is not None:
        db_path_t2 = db_path_1900
    if use_umap is not None:
        clustering_reduction = 'umap' if use_umap else None
    if umap_n_components is not None:
        clustering_n_components = umap_n_components

    # Build config
    config = AnalysisConfig(
        project_id=project_id,
        target_word=target_word,
        db_path_t1=db_path_t1,
        db_path_t2=db_path_t2,
        period_t1_label=period_t1_label,
        period_t2_label=period_t2_label,
        model_name=model_name,
        n_samples=n_samples,
        k_neighbors=k_neighbors,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        wsi_algorithm=wsi_algorithm,
        pos_filter=pos_filter,
        clustering_reduction=clustering_reduction,
        clustering_n_components=clustering_n_components,
        viz_reduction=viz_reduction,
        viz_max_instances=viz_max_instances,
        context_window=context_window
    )

    # Validate
    error = validate_config(config)
    if error:
        print(f"Error: {error}. Please run ingestion first.")
        return

    # Step 1: Clean output
    print("--- Cleaning up old visualizations ---")
    clean_output_directory(config.output_dir)

    # Step 2: Initialize vector store and fetch embeddings
    print("--- Checking Vector Store ---")

    if not config.project_id:
        print("Error: project_id is required.")
        return

    try:
        vector_store = VectorStore(persistence_path="data/chroma_db")
    except Exception as e:
        print(f"Could not initialize Vector Store: {e}")
        return

    # Get collection names for this project and model
    safe_model = config.model_name.replace("/", "_").replace("-", "_")
    coll_t1, coll_t2 = vector_store.get_collection_names_for_model(config.project_id, safe_model)
    if coll_t1 is None and coll_t2 is None:
        print(f"No embedding collections found for project '{config.project_id}' and model '{config.model_name}'.")
        print("Available models for this project:", vector_store.get_available_models(config.project_id))
        return

    data_t1 = None
    if coll_t1:
        data_t1 = fetch_embeddings_from_store(
            vector_store, coll_t1, config.db_path_t1,
            config.target_word, config.n_samples,
            config.pos_filter, config.context_window,
            config.period_t1_label
        )

    data_t2 = None
    if coll_t2:
        data_t2 = fetch_embeddings_from_store(
            vector_store, coll_t2, config.db_path_t2,
            config.target_word, config.n_samples,
            config.pos_filter, config.context_window,
            config.period_t2_label
        )

    if (data_t1 is None or data_t1.is_empty()) and (data_t2 is None or data_t2.is_empty()):
        print("No embeddings found in Vector Store. Please run embedding generation first.")
        return

    # Handle case where only one period has data
    if data_t1 is None or data_t1.is_empty():
        data_t1 = EmbeddingData(
            embeddings=np.empty((0, data_t2.embeddings.shape[1])),
            sentences=[], spans=[], filenames=[],
            period_label=config.period_t1_label
        )
    if data_t2 is None or data_t2.is_empty():
        data_t2 = EmbeddingData(
            embeddings=np.empty((0, data_t1.embeddings.shape[1])),
            sentences=[], spans=[], filenames=[],
            period_label=config.period_t2_label
        )

    # Step 3: Combine embeddings
    combined = combine_embedding_data(data_t1, data_t2)

    # Step 4: Pre-clustering dimensionality reduction
    if config.clustering_reduction:
        print(f"--- Pre-clustering Reduction: {config.clustering_reduction.upper()} "
              f"(n={config.clustering_n_components}) ---")

    clustering_embeddings = apply_dimensionality_reduction(
        combined.embeddings,
        config.clustering_reduction,
        config.clustering_n_components
    )

    # Step 5: Run clustering
    print(f"--- Running WSI ({config.wsi_algorithm.upper()}) ---")
    sense_labels = run_word_sense_induction(
        clustering_embeddings,
        config.wsi_algorithm,
        config.min_cluster_size,
        config.n_clusters
    )

    # Step 6: Prepare visualization data
    print(f"--- Visualizing (reduction: {config.viz_reduction.upper()}) ---")

    sub_embs, sub_sense_labs, sub_sents, sub_fnames, sub_spans, sub_indices = \
        subsample_for_visualization(
            combined.embeddings,
            sense_labels,
            combined.sentences,
            combined.filenames,
            config.viz_max_instances,
            combined.highlight_spans
        )

    sub_time_labs = combined.time_labels[sub_indices]

    # Compute 2D projection
    print("Computing 2D projection...")
    viz = Visualizer(method=config.viz_reduction)
    embeddings_2d = viz.fit_transform(sub_embs)

    viz_data = VisualizationData(
        embeddings_2d=embeddings_2d,
        sense_labels=sub_sense_labs,
        time_labels=sub_time_labs,
        sentences=sub_sents,
        filenames=sub_fnames,
        highlight_spans=sub_spans,
        original_indices=sub_indices
    )

    # Step 7: Create visualizations
    create_visualizations(
        viz_data, config.target_word, config.output_dir, config.viz_reduction
    )

    # Step 8: Create neighbor plots
    print("Plotting Neighbors for Sense Clusters...")
    create_neighbor_plots(
        combined.embeddings,
        sense_labels,
        config.target_word,
        vector_store,
        coll_t1,
        coll_t2,
        config.k_neighbors,
        config.output_dir
    )

    print("Done!")


# =============================================================================
# CLI Entry Point
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Semantic Change Analysis for a single word."
    )
    parser.add_argument(
        "--word", type=str, default="current",
        help="Target word to analyze"
    )
    parser.add_argument(
        "--db-t1", type=str, default="data/corpus_t1.db",
        help="Path to first period DB"
    )
    parser.add_argument(
        "--db-t2", type=str, default="data/corpus_t2.db",
        help="Path to second period DB"
    )
    parser.add_argument(
        "--label-t1", type=str, default="1800",
        help="Label for first time period"
    )
    parser.add_argument(
        "--label-t2", type=str, default="1900",
        help="Label for second time period"
    )
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--pos", type=str, default=None,
        help="POS filter (e.g., NOUN, VERB)"
    )
    parser.add_argument(
        "--clustering-reduction", type=str, default=None,
        choices=[None, 'pca', 'umap', 'tsne'],
        help="Dimensionality reduction before clustering"
    )
    parser.add_argument(
        "--clustering-dims", type=int, default=50,
        help="Number of dimensions for pre-clustering reduction"
    )
    parser.add_argument(
        "--viz-reduction", type=str, default='pca',
        choices=['pca', 'umap', 'tsne'],
        help="Dimensionality reduction for visualization"
    )
    parser.add_argument(
        "--n-top-sents", type=int, default=10,
        help="Number of sentences for contextual MLM (fallback)"
    )
    parser.add_argument(
        "--k-per-sent", type=int, default=6,
        help="Predictions per sentence for contextual MLM (fallback)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_single_analysis(
        target_word=args.word,
        db_path_t1=args.db_t1,
        db_path_t2=args.db_t2,
        period_t1_label=args.label_t1,
        period_t2_label=args.label_t2,
        model_name=args.model,
        pos_filter=args.pos,
        clustering_reduction=args.clustering_reduction,
        clustering_n_components=args.clustering_dims,
        viz_reduction=args.viz_reduction,
        n_top_sentences=args.n_top_sents,
        k_per_sentence=args.k_per_sent
    )
