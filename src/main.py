import os
import shutil
import glob
import numpy as np
import argparse
import sqlite3
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict, Counter
from semantic_change.corpus import CorpusManager
from semantic_change.embedding import BertEmbedder
from semantic_change.wsi import WordSenseInductor
from semantic_change.visualization import Visualizer
from semantic_change.vector_store import VectorStore

def run_single_analysis(
    target_word="current",
    db_path_t1="data/corpus_t1.db",
    db_path_t2="data/corpus_t2.db",
    period_t1_label="1800",
    period_t2_label="1900",
    model_name="bert-base-uncased",
    k_neighbors=10,
    min_cluster_size=3,
    n_clusters=3,
    wsi_algorithm="hdbscan",
    pos_filter=None,
    embedder=None,
    # Pre-clustering dimensionality reduction
    clustering_reduction=None,  # None, 'pca', or 'umap'
    clustering_n_components=50,
    # Visualization dimensionality reduction
    viz_reduction='pca',  # 'pca', 'tsne', or 'umap'
    # Legacy parameters (for backwards compatibility)
    use_umap=None,  # Deprecated, use clustering_reduction instead
    umap_n_components=None,  # Deprecated, use clustering_n_components instead
    db_path_1800=None,  # Deprecated, use db_path_t1
    db_path_1900=None,  # Deprecated, use db_path_t2
    n_samples=50,
    viz_max_instances=100,
    context_window=0
):
    OUTPUT_DIR = "output"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_chroma_neighbors(
    vector_store: VectorStore,
    collection_name: str,
    centroid: np.ndarray,
    target_word: str = None,
    n_neighbors: int = 10,
    max_candidates: int = 500
) -> Dict[str, np.ndarray]:
    """
    Finds nearest neighbors by iteratively querying the vector store.
    Handles high-frequency word dominance by querying in batches until 
    enough unique lemmas are found.
    """
    
    unique_neighbors = {} # lemma -> vector
    candidates_seen = 0
    batch_size = n_neighbors * 5
    
    # Prepare skip list
    skip_lemmas = set()
    if target_word:
        skip_lemmas.add(target_word.lower())
    
    # We can't easily "exclude" IDs in Chroma without a complex filter.
    # Instead, we just query a large batch. If that's not enough, we query more?
    # Chroma's query doesn't support offset.
    # So we simply query a large enough number. 
    # If the user wants 10 neighbors, querying 100 or 200 should usually be enough.
    # If a word is extremely frequent (e.g. 1000 times), we might miss others.
    # Ideally, we'd use a where clause to exclude already found words, but Chroma 
    # 'where' clause on metadata 'lemma' $ne '...' might be inefficient if list is long.
    
    # Strategy: Query a large batch (e.g. 500).
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
    vecs = results['embeddings'][0] if 'embeddings' in results and results['embeddings'] else []
    
    # Tally up
    lemma_counts = Counter()
    lemma_vectors_sum = defaultdict(lambda: np.zeros(centroid.shape))
    
    for i, m in enumerate(metas):
        lemma = m.get('lemma', '').lower()
        if not lemma or len(lemma) < 2: continue
        if lemma in skip_lemmas: continue
        
        lemma_counts[lemma] += 1
        
        if vecs:
            v = np.array(vecs[i])
            lemma_vectors_sum[lemma] += v
            
    # Get top frequent unique lemmas from this batch
    # Note: "frequent" in the neighborhood means "dense" which is good.
    # But we want the top N *closest*? 
    # Chroma returns them sorted by distance.
    # So the first appearance of a lemma is its closest instance.
    # We should probably prioritize by "first appearance rank" rather than "count in top 500".
    # Let's just take the unique lemmas in order of appearance in the results.
    
    top_unique = []
    seen = set()
    for i, m in enumerate(metas):
        lemma = m.get('lemma', '').lower()
        if not lemma or len(lemma) < 2: continue
        if lemma in skip_lemmas: continue
        if lemma in seen: continue
        
        seen.add(lemma)
        
        # Calculate mean vector for this lemma (using all instances found so far? 
        # Or just this one? Using mean of all instances in the neighborhood is cleaner)
        # We computed sum and count above.
        mean_vec = lemma_vectors_sum[lemma] / lemma_counts[lemma]
        top_unique.append((lemma, mean_vec))
        
        if len(top_unique) >= n_neighbors:
            break
            
    return {lemma: vec for lemma, vec in top_unique}

def run_single_analysis(
    target_word="current",
    db_path_t1="data/corpus_t1.db",
    db_path_t2="data/corpus_t2.db",
    period_t1_label="1800",
    period_t2_label="1900",
    model_name="bert-base-uncased",
    k_neighbors=10,
    min_cluster_size=3,
    n_clusters=3,
    wsi_algorithm="hdbscan",
    pos_filter=None,
    embedder=None,
    # Pre-clustering dimensionality reduction
    clustering_reduction=None,  # None, 'pca', or 'umap'
    clustering_n_components=50,
    # Visualization dimensionality reduction
    viz_reduction='pca',  # 'pca', 'tsne', or 'umap'
    # Legacy parameters (for backwards compatibility)
    use_umap=None,  # Deprecated, use clustering_reduction instead
    umap_n_components=None,  # Deprecated, use clustering_n_components instead
    db_path_1800=None,  # Deprecated, use db_path_t1
    db_path_1900=None,  # Deprecated, use db_path_t2
    n_samples=50,
    viz_max_instances=100,
    context_window=0,
    n_top_sentences=10, # Re-added for signature compatibility
    k_per_sentence=6    # Re-added for signature compatibility
):
    OUTPUT_DIR = "output"

    print("--- Cleaning up old visualizations ---")
    # ... cleanup logic ... (omitted for brevity in replacement context but kept in code)
    for f in glob.glob(os.path.join(OUTPUT_DIR, "neighbors_cluster_*.html")):
        try: os.remove(f)
        except: pass
    for f in ["sense_clusters.html", "time_period.html", "sense_time_combined.html"]:
        f_path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(f_path):
            try: os.remove(f_path)
            except: pass

    # Handle legacy parameter names for backwards compatibility
    if db_path_1800 is not None:
        db_path_t1 = db_path_1800
    if db_path_1900 is not None:
        db_path_t2 = db_path_1900

    # 1. Corpus Management
    print("--- Initializing Corpus ---")
    manager = CorpusManager()

    if not os.path.exists(db_path_t1) or not os.path.exists(db_path_t2):
        print("Error: Ingested data not found. Please run 'run_ingest.py' first.")
        return

    # Register corpora with user-defined labels
    manager.add_corpus(period_t1_label, os.path.dirname(db_path_t1), db_path_t1)
    manager.add_corpus(period_t2_label, os.path.dirname(db_path_t2), db_path_t2)

    # 2. Check Vector Store or Compute Embeddings
    print("--- Checking Vector Store ---")

    def get_collection_name(period_id, model):
        safe_model = model.replace("/", "_").replace("-", "_")
        # Use t1/t2 explicitly for internal storage, not the user label
        # But we need to map period_t1_label to "t1". 
        # Since we just have two variables, we can hardcode the calls.
        return f"embeddings_{period_id}_{safe_model}"

    # Use "t1" and "t2" explicitly as defined in run_batch_analysis.py
    coll_t1 = get_collection_name("t1", model_name)
    coll_t2 = get_collection_name("t2", model_name)
    
    # Hold reference to vector_store for later neighbor query
    v_store_ref = None
    
    try:
        v_store = VectorStore(persistence_path="data/chroma_db")
        v_store_ref = v_store
        
        def fetch_from_store(coll_name, db_path):
            try:
                # Note: We only fetch from store if context_window is 0,
                # because cached embeddings are usually sentence-level.
                if context_window > 0:
                    return None, None, None

                word_lower = target_word.lower()
                data = v_store.get_by_metadata(coll_name, where={"lemma": word_lower}, limit=n_samples)
                # Fix: explicit check for list length
                if data and data.get('embeddings') is not None and len(data['embeddings']) > 0:
                    print(f"Found {len(data['embeddings'])} cached embeddings in {coll_name}")
                    # Fetch sentences from SQLite, char offsets are in metadata
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    sents = []
                    filenames = []
                    spans = []
                    for m in data['metadatas']:
                        sent_id = m['sentence_id']
                        # Get sentence text and filename
                        # We need to join with files table
                        try:
                            row = cursor.execute("""
                                SELECT s.text, f.filename 
                                FROM sentences s 
                                JOIN files f ON s.file_id = f.id 
                                WHERE s.id=?
                            """, (sent_id,)).fetchone()
                            
                            if row:
                                sents.append(row[0])
                                filenames.append(row[1])
                            else:
                                sents.append("[Missing Sentence]")
                                filenames.append("Unknown")
                        except sqlite3.OperationalError:
                            # Fallback if files table join fails (e.g. old DB schema?)
                            row = cursor.execute("SELECT text FROM sentences WHERE id=?", (sent_id,)).fetchone()
                            sents.append(row[0] if row else "[Missing Sentence]")
                            filenames.append("Unknown")

                        # Get char offsets directly from metadata (stored by batch analysis)
                        start_char = m.get('start_char')
                        end_char = m.get('end_char')
                        if start_char is not None and end_char is not None:
                            spans.append((start_char, end_char))
                        else:
                            spans.append(None)
                    conn.close()
                    return np.array(data['embeddings']), sents, spans, filenames
            except Exception as e:
                print(f"Warning: fetch_from_store failed: {e}")
            return None, None, None, None

        emb_t1, valid_sents_t1, spans_t1, filenames_t1 = fetch_from_store(coll_t1, db_path_t1)
        emb_t2, valid_sents_t2, spans_t2, filenames_t2 = fetch_from_store(coll_t2, db_path_t2)
    except Exception as e:
        print(f"Could not initialize Vector Store: {e}")
        emb_t1, emb_t2 = None, None
        spans_t1, spans_t2 = None, None
        filenames_t1, filenames_t2 = None, None

    # Helper for dynamic context
    def prepare_batch_items(samples, window):
        batch_items = []
        # Cache for loaded files to avoid re-reading the same file many times in one run
        file_cache = {}

        for s in samples:
            text = s['sentence']
            start_char = s['start_char']
            
            if window > 0 and s.get('file_path'):
                fpath = s['file_path']
                if fpath not in file_cache:
                    try:
                        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                            file_cache[fpath] = f.read()
                    except:
                        file_cache[fpath] = None
                
                fcontent = file_cache[fpath]
                if fcontent:
                    # target index in file
                    target_idx = s['file_offset_start'] + s['start_char']
                    extract_start = max(0, target_idx - window)
                    extract_end = min(len(fcontent), target_idx + len(s['matched_word']) + window)
                    
                    text = fcontent[extract_start : extract_end]
                    start_char = target_idx - extract_start
            
            end_char = start_char + len(s['matched_word'])
            batch_items.append({
                'id': s['sentence_id'],
                'text': text,
                'targets': [(s['lemma'], start_char, end_char)]
            })
        return batch_items

    # If missing from cache, fallback to on-the-fly extraction
    if emb_t1 is None or len(emb_t1) == 0:
        print(f"--- Querying samples for '{target_word}' (POS: {pos_filter if pos_filter else 'ALL'}, n={n_samples}) ---")
        samples_t1 = manager.get_corpus(period_t1_label).query_samples(target_word, n=n_samples, pos_filter=pos_filter)

        if embedder is None:
            embedder = BertEmbedder(model_name=model_name)

        print(f"--- Generating Embeddings {period_t1_label} ({model_name}, window={context_window}) ---")
        batch_items_t1 = prepare_batch_items(samples_t1, context_window)
        extracted = embedder.batch_extract(batch_items_t1, batch_size=32)

        emb_t1 = np.array([x['vector'] for x in extracted])
        id_to_text = {x['id']: x['text'] for x in batch_items_t1}
        # Create map from sentence_id to filename (basename of file_path)
        id_to_filename = {s['sentence_id']: os.path.basename(s['file_path']) if s.get('file_path') else "Unknown" for s in samples_t1}
        
        valid_sents_t1 = [id_to_text[x['sentence_id']] for x in extracted]
        filenames_t1 = [id_to_filename.get(x['sentence_id'], "Unknown") for x in extracted]
        # Track word positions for highlighting
        spans_t1 = [(x['start_char'], x['end_char']) for x in extracted]

    if emb_t2 is None or len(emb_t2) == 0:
        if 'samples_t2' not in locals():
            samples_t2 = manager.get_corpus(period_t2_label).query_samples(target_word, n=n_samples, pos_filter=pos_filter)

        if embedder is None:
            embedder = BertEmbedder(model_name=model_name)

        print(f"--- Generating Embeddings {period_t2_label} ({model_name}, window={context_window}) ---")
        batch_items_t2 = prepare_batch_items(samples_t2, context_window)
        extracted = embedder.batch_extract(batch_items_t2, batch_size=32)

        emb_t2 = np.array([x['vector'] for x in extracted])
        id_to_text = {x['id']: x['text'] for x in batch_items_t2}
        id_to_filename = {s['sentence_id']: os.path.basename(s['file_path']) if s.get('file_path') else "Unknown" for s in samples_t2}
        
        valid_sents_t2 = [id_to_text[x['sentence_id']] for x in extracted]
        filenames_t2 = [id_to_filename.get(x['sentence_id'], "Unknown") for x in extracted]
        # Track word positions for highlighting
        spans_t2 = [(x['start_char'], x['end_char']) for x in extracted]

    if (emb_t1 is None or len(emb_t1) == 0) and (emb_t2 is None or len(emb_t2) == 0):
        print("No embeddings found. Exiting.")
        return

    # Combine for visualization
    all_embeddings = np.vstack([emb_t1, emb_t2])
    time_labels = np.array([period_t1_label] * len(emb_t1) + [period_t2_label] * len(emb_t2))
    all_sentences = np.array(valid_sents_t1 + valid_sents_t2)
    all_filenames = np.array((filenames_t1 or []) + (filenames_t2 or []))

    # Combine highlight spans (None if from cache without span info)
    if spans_t1 is not None and spans_t2 is not None:
        all_highlight_spans = spans_t1 + spans_t2
    elif spans_t1 is not None:
        all_highlight_spans = spans_t1 + [None] * len(emb_t2)
    elif spans_t2 is not None:
        all_highlight_spans = [None] * len(emb_t1) + spans_t2
    else:
        all_highlight_spans = None
    
    # 3. Word Sense Induction
    clustering_embeddings = all_embeddings

    # Handle legacy parameters for backwards compatibility
    if use_umap is not None:
        clustering_reduction = 'umap' if use_umap else None
    if umap_n_components is not None:
        clustering_n_components = umap_n_components

    # Pre-clustering dimensionality reduction (optional)
    if clustering_reduction:
        n_components = clustering_n_components
        if len(all_embeddings) <= n_components:
            print(f"Warning: Not enough data ({len(all_embeddings)}) for {n_components} components, skipping reduction.")
        else:
            print(f"--- Pre-clustering Reduction: {clustering_reduction.upper()} (n={n_components}) ---")
            if clustering_reduction.lower() == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=n_components, random_state=42)
                clustering_embeddings = reducer.fit_transform(all_embeddings)
            elif clustering_reduction.lower() == 'umap':
                try:
                    import umap
                    reducer = umap.UMAP(n_components=n_components, random_state=42)
                    clustering_embeddings = reducer.fit_transform(all_embeddings)
                except ImportError:
                    print("Warning: umap-learn not installed. Skipping reduction.")
            elif clustering_reduction.lower() == 'tsne':
                from sklearn.manifold import TSNE
                # t-SNE typically needs perplexity < n_samples
                perplexity = min(30, len(all_embeddings) - 1)
                reducer = TSNE(n_components=min(n_components, 3), perplexity=perplexity, random_state=42)
                clustering_embeddings = reducer.fit_transform(all_embeddings)
            else:
                print(f"Warning: Unknown reduction method '{clustering_reduction}', skipping.")
    
    print(f"--- Running WSI ({wsi_algorithm.upper()}) ---")
    wsi = WordSenseInductor(
        algorithm=wsi_algorithm, 
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters
    ) 
    sense_labels = wsi.fit_predict(clustering_embeddings)
    
    # Helper to limit visualization data
    def subsample_for_viz(embs, labs, sents, fnames, max_per_class, spans=None, return_indices=False):
        unique_labs = np.unique(labs)
        indices_to_keep = []
        for l in unique_labs:
            l_indices = np.where(labs == l)[0]
            if len(l_indices) > max_per_class:
                # Randomly select max_per_class
                selected = np.random.choice(l_indices, max_per_class, replace=False)
                indices_to_keep.extend(selected)
            else:
                indices_to_keep.extend(l_indices)

        sub_spans = None
        if spans is not None:
            sub_spans = [spans[i] for i in indices_to_keep]

        if return_indices:
            return embs[indices_to_keep], labs[indices_to_keep], sents[indices_to_keep], fnames[indices_to_keep], sub_spans, indices_to_keep
        return embs[indices_to_keep], labs[indices_to_keep], sents[indices_to_keep], fnames[indices_to_keep], sub_spans

    # 4. Visualization
    print(f"--- Visualizing (reduction: {viz_reduction.upper()}) ---")
    viz = Visualizer(method=viz_reduction)

    # Subsample ONCE for all plots to ensure consistency
    # We use sense_labels for balancing to ensure all clusters are represented
    viz_embs_raw, viz_sense_labs, viz_sents, viz_fnames, viz_spans, viz_indices = subsample_for_viz(
        all_embeddings, sense_labels, all_sentences, all_filenames, viz_max_instances, all_highlight_spans,
        return_indices=True
    )
    viz_time_labs = time_labels[viz_indices]
    
    # Compute dimensionality reduction ONCE
    print("Computing 2D projection...")
    viz_2d = viz.fit_transform(viz_embs_raw)

    print("Plotting Clustering by Time...")
    viz.plot_clustering(
        viz_2d,
        labels=viz_time_labs,
        sentences=viz_sents,
        filenames=viz_fnames,
        title=f"'{target_word}' by Time Period",
        save_path=os.path.join(OUTPUT_DIR, "time_period.html"),
        highlight_spans=viz_spans
    )

    print("Plotting Clustering by Sense...")
    viz.plot_clustering(
        viz_2d,
        labels=viz_sense_labs,
        sentences=viz_sents,
        filenames=viz_fnames,
        title=f"'{target_word}' by Sense Cluster",
        save_path=os.path.join(OUTPUT_DIR, "sense_clusters.html"),
        highlight_spans=viz_spans
    )

    print("Plotting Combined Sense × Time...")
    viz.plot_combined_clustering(
        viz_2d,
        sense_labels=viz_sense_labs,
        time_labels=list(viz_time_labs),
        sentences=viz_sents,
        filenames=viz_fnames,
        title=f"'{target_word}' by Sense × Time",
        save_path=os.path.join(OUTPUT_DIR, "sense_time_combined.html"),
        highlight_spans=viz_spans
    )

    # 5. Neighbor Plot
    print("Plotting Neighbors for Sense Clusters...")
    unique_clusters = sorted(list(set(sense_labels)))
    
    # Ensure embedder is available for neighbor projection (fallback)
    if embedder is None:
        embedder = BertEmbedder(model_name=model_name)

    for cluster_id in unique_clusters:
        mask = (sense_labels == cluster_id)
        cluster_embs = all_embeddings[mask]
        if len(cluster_embs) == 0: continue
        
        centroid = np.mean(cluster_embs, axis=0)
        
        # Try to use Vector Store first
        neighbors = {}
        source_label = "ChromaDB"
        
        if v_store_ref:
            try:
                print(f"Querying ChromaDB for cluster {cluster_id} neighbors...")
                n1 = get_chroma_neighbors(v_store_ref, coll_t1, centroid, target_word=target_word, n_neighbors=k_neighbors)
                n2 = get_chroma_neighbors(v_store_ref, coll_t2, centroid, target_word=target_word, n_neighbors=k_neighbors)
                
                # Merge logic: if word in both, take average or closest. 
                # Simple merge:
                merged = {**n1, **n2} 
                # If we have too many, sort by distance to centroid.
                # Since we only have vectors in the map, we need to recalc distance or just trust the initial top k.
                # Let's simple slice.
                neighbors = dict(list(merged.items())[:k_neighbors*2])
            except Exception as e:
                print(f"ChromaDB lookup failed: {e}")
                neighbors = {}
        
        # Fallback to MLM if Chroma returned nothing (e.g. store is empty)
        if not neighbors:
            source_label = "MLM Projection"
            print(f"Falling back to MLM projection for cluster {cluster_id}")
            neighbors = embedder.get_nearest_neighbors(
                centroid, 
                target_word=target_word, 
                k=k_neighbors, 
                pos_filter=pos_filter
            )
        
        print(f"Cluster {cluster_id} Neighbors ({source_label}): {list(neighbors.keys())}")
        
        viz.plot_neighbors(
            centroid, 
            neighbors, 
            centroid_name=target_word,
            title=f"Semantic Neighbors for Cluster {cluster_id} ({source_label})",
            save_path=os.path.join(OUTPUT_DIR, f"neighbors_cluster_{cluster_id}.html")
        )
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Semantic Change Analysis for a single word.")
    parser.add_argument("--word", type=str, default="current", help="Target word to analyze")
    parser.add_argument("--db-t1", type=str, default="data/corpus_t1.db", help="Path to first period DB")
    parser.add_argument("--db-t2", type=str, default="data/corpus_t2.db", help="Path to second period DB")
    parser.add_argument("--label-t1", type=str, default="1800", help="Label for first time period")
    parser.add_argument("--label-t2", type=str, default="1900", help="Label for second time period")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--pos", type=str, default=None, help="POS filter (e.g., NOUN, VERB)")
    parser.add_argument("--clustering-reduction", type=str, default=None,
                        choices=[None, 'pca', 'umap', 'tsne'],
                        help="Dimensionality reduction before clustering (None, pca, umap, tsne)")
    parser.add_argument("--clustering-dims", type=int, default=50,
                        help="Number of dimensions for pre-clustering reduction")
    parser.add_argument("--viz-reduction", type=str, default='pca',
                        choices=['pca', 'umap', 'tsne'],
                        help="Dimensionality reduction for visualization (pca, umap, tsne)")
    parser.add_argument("--n-top-sents", type=int, default=10, help="Number of sentences to use for contextual MLM (fallback)")
    parser.add_argument("--k-per-sent", type=int, default=6, help="Predictions per sentence for contextual MLM (fallback)")

    args = parser.parse_args()

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
    