import sqlite3
import torch
import numpy as np
import os
import time
import argparse
import glob as globlib
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Tuple, Set, Any, Union, Type, Optional
from .embedding import BertEmbedder, detect_optimal_batch_size
from .vector_store import VectorStore
from .corpus import get_spacy_model_from_db
from tqdm import tqdm
from tqdm.std import tqdm as tqdm_std

# Default staging directory for NPZ files
DEFAULT_STAGING_DIR = "data/embedding_staging"

def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def get_frequent_words(db_path: str, min_freq: int = 25, pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV')) -> List[str]:
    """
    Finds words in a specific database with frequency >= min_freq for the given POS tags.
    """
    print(f"Finding frequent content words in {os.path.basename(db_path)} (min_freq={min_freq})...")
    conn = get_db_connection(db_path)
    
    try:
        if isinstance(pos_filter, (list, tuple)):
            placeholders = ','.join('?' for _ in pos_filter)
            query = f"SELECT lemma, count(*) as c FROM tokens WHERE pos IN ({placeholders}) GROUP BY lemma HAVING c >= {min_freq}"
            rows = conn.execute(query, pos_filter).fetchall()
        else:
            query = f"SELECT lemma, count(*) as c FROM tokens WHERE pos=? GROUP BY lemma HAVING c >= {min_freq}"
            rows = conn.execute(query, (pos_filter,)).fetchall()
    except sqlite3.OperationalError as e:
        print(f"Error querying {db_path}: {e}")
        return []
    finally:
        conn.close()
    
    words = {row[0] for row in rows}
    
    # Filter vocabulary
    clean_words = []
    
    # Load stop words
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        stop_words = nlp.Defaults.stop_words
    except:
        stop_words = set()

    for w in words:
        if len(w) < 2: continue 
        if not w.replace('-', '').isalpha(): continue 
        if w in stop_words: continue
        
        clean_words.append(w)
        
    print(f"Found {len(clean_words)} valid words in {os.path.basename(db_path)}.")
    return sorted(list(clean_words))

def get_shared_words(db_path_1: str, db_path_2: str, min_freq: int = 25, pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV')) -> List[str]:
    """Finds words that appear frequently in both databases."""
    words_1 = set(get_frequent_words(db_path_1, min_freq, pos_filter))
    words_2 = set(get_frequent_words(db_path_2, min_freq, pos_filter))
    shared = words_1.intersection(words_2)
    return sorted(list(shared))

def collect_sentences_for_words(db_path: str, target_words: List[str], max_samples: int = 500,
                                pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'),
                                tqdm_class: Type[tqdm_std] = tqdm) -> List[Dict[str, Any]]:
    """
    Retrieves sentences containing the target words from the SQLite database.
    Optimized to fetch in batches to avoid N+1 query problem.

    Args:
        tqdm_class: Progress bar class to use (tqdm for CLI, stqdm for Streamlit).
    """
    print(f"Collecting sentences from {db_path} (max_samples={max_samples}, pos={pos_filter})...")
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    sentence_tasks = defaultdict(list)
    unique_sentence_ids = set()

    # Build query parts for POS filtering
    if isinstance(pos_filter, (list, tuple)):
        pos_condition = f"pos IN ({','.join('?' for _ in pos_filter)})"
        pos_params = list(pos_filter)
    else:
        pos_condition = "pos = ?"
        pos_params = [pos_filter]

    # Process words in chunks to keep SQL query size reasonable
    word_chunk_size = 100
    total_chunks = (len(target_words) + word_chunk_size - 1) // word_chunk_size

    with tqdm_class(total=total_chunks, desc="Preparing word samples") as pbar:
        for i in range(0, len(target_words), word_chunk_size):
            chunk_words = target_words[i : i + word_chunk_size]
            placeholders = ','.join('?' for _ in chunk_words)

            # We need a way to limit per word in SQL, or we fetch all and limit in Python.
            # Window functions would be ideal (ROW_NUMBER() OVER (PARTITION BY lemma)) but require SQLite 3.25+

            try:
                query = f"""
                    SELECT sentence_id, start_char, text, pos, lemma
                    FROM (
                        SELECT
                            sentence_id, start_char, text, pos, lemma,
                            ROW_NUMBER() OVER (PARTITION BY lemma ORDER BY random()) as rn
                        FROM tokens
                        WHERE lemma IN ({placeholders}) AND {pos_condition}
                    )
                    WHERE rn <= ?
                """
                params = chunk_words + pos_params + [max_samples]
                rows = cursor.execute(query, params).fetchall()

                for sent_id, start_char, token_text, token_pos, lemma in rows:
                    unique_sentence_ids.add(sent_id)
                    end_char = start_char + len(token_text)
                    sentence_tasks[sent_id].append((lemma, token_pos, start_char, end_char))

            except sqlite3.OperationalError:
                # Fallback for older SQLite versions: Fetch all and limit in Python
                query = f"SELECT sentence_id, start_char, text, pos, lemma FROM tokens WHERE lemma IN ({placeholders}) AND {pos_condition}"
                params = chunk_words + pos_params
                rows = cursor.execute(query, params).fetchall()

                # Group by lemma to limit
                counts = defaultdict(int)
                import random
                random.shuffle(rows)

                for sent_id, start_char, token_text, token_pos, lemma in rows:
                    if counts[lemma] < max_samples:
                        unique_sentence_ids.add(sent_id)
                        end_char = start_char + len(token_text)
                        sentence_tasks[sent_id].append((lemma, token_pos, start_char, end_char))
                        counts[lemma] += 1

            pbar.update(1)

    sent_ids_list = list(unique_sentence_ids)
    chunk_size = 999
    total_text_chunks = (len(sent_ids_list) + chunk_size - 1) // chunk_size

    batch_items = []

    with tqdm_class(total=total_text_chunks, desc="Fetching texts") as pbar:
        for start_idx in range(0, len(sent_ids_list), chunk_size):
            chunk = sent_ids_list[start_idx : start_idx+chunk_size]
            placeholders = ','.join('?' for _ in chunk)
            rows = cursor.execute(f"SELECT id, text FROM sentences WHERE id IN ({placeholders})", chunk).fetchall()

            for sid, text in rows:
                targets = sentence_tasks.get(sid, [])
                if targets:
                    batch_items.append({
                        "id": sid,
                        "text": text,
                        "targets": targets
                    })

            pbar.update(1)

    conn.close()
    return batch_items

def process_corpus(db_path: str, collection_name: str, target_words: List[str],
                   embedder: BertEmbedder, vector_store: VectorStore, max_samples: int = 200,
                   pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'),
                   embed_batch_size: int = None, tqdm_class: Type[tqdm_std] = tqdm):
    """
    Orchestrates the processing of a single corpus: collection, embedding extraction, and saving to ChromaDB.
    Uses async DB writes to overlap with GPU computation.

    Args:
        embed_batch_size: Batch size for embedding extraction. If None, auto-detects based on GPU VRAM.
        tqdm_class: Progress bar class to use (tqdm for CLI, stqdm for Streamlit).
    """
    batch_items = collect_sentences_for_words(db_path, target_words, max_samples=max_samples,
                                               pos_filter=pos_filter, tqdm_class=tqdm_class)
    total_sents = len(batch_items)
    if total_sents == 0:
        print(f"No sentences found for {os.path.basename(db_path)}.")
        return

    # Increased chunk size to reduce ChromaDB write frequency (avoids compaction errors)
    chunk_size = 200
    # Auto-detect optimal batch size based on GPU VRAM if not specified
    if embed_batch_size is None:
        embed_batch_size = detect_optimal_batch_size()
        print(f"  Auto-detected embedding batch size: {embed_batch_size}")
    desc = f"Processing {os.path.basename(db_path)} -> ChromaDB"

    def save_chunk(chunk_results):
        if not chunk_results:
            return
        vector_store.add_embeddings(
            collection_name=collection_name,
            embeddings=[r['vector'] for r in chunk_results],
            metadatas=[{
                "lemma": r['lemma'],
                "pos": r['pos'],
                "sentence_id": r['sentence_id'],
                "start_char": r['start_char'],
                "end_char": r['end_char'],
                "token": r.get('token', r['lemma'])  # Store actual token text for exact match queries
            } for r in chunk_results],
            ids=[f"{collection_name}_{r['lemma']}_{r['pos']}_{r['sentence_id']}_{r['start_char']}" for r in chunk_results]
        )

    # Use thread pool to overlap DB writes with GPU computation
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_future: Future = None

        with tqdm_class(total=total_sents, desc=desc) as pbar:
            for i in range(0, total_sents, chunk_size):
                chunk = batch_items[i : i + chunk_size]
                chunk_results = embedder.batch_extract(chunk, batch_size=embed_batch_size)

                # Wait for previous write to complete before submitting new one
                if pending_future is not None:
                    pending_future.result()

                # Submit DB write asynchronously
                pending_future = executor.submit(save_chunk, chunk_results)
                pbar.update(len(chunk))

            # Wait for final write
            if pending_future is not None:
                pending_future.result()


def process_corpus_staged(db_path: str, collection_name: str, target_words: List[str],
                          embedder: BertEmbedder, staging_dir: str, max_samples: int = 200,
                          pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'),
                          embed_batch_size: int = None, tqdm_class: Type[tqdm_std] = tqdm) -> int:
    """
    Process corpus and write embeddings to NPZ files instead of ChromaDB.
    This is much faster than direct ChromaDB writes due to sequential I/O.

    Returns the number of embeddings written.
    """
    batch_items = collect_sentences_for_words(db_path, target_words, max_samples=max_samples,
                                               pos_filter=pos_filter, tqdm_class=tqdm_class)
    total_sents = len(batch_items)
    if total_sents == 0:
        print(f"No sentences found for {os.path.basename(db_path)}.")
        return 0

    # Create staging directory for this collection
    collection_staging_dir = os.path.join(staging_dir, collection_name)
    os.makedirs(collection_staging_dir, exist_ok=True)

    # Larger chunk size for NPZ - we can afford bigger batches since writes are fast
    chunk_size = 500
    if embed_batch_size is None:
        embed_batch_size = detect_optimal_batch_size()
        print(f"  Auto-detected embedding batch size: {embed_batch_size}")

    desc = f"Processing {os.path.basename(db_path)} -> NPZ"
    total_written = 0
    chunk_idx = 0

    with tqdm_class(total=total_sents, desc=desc) as pbar:
        for i in range(0, total_sents, chunk_size):
            chunk = batch_items[i : i + chunk_size]
            chunk_results = embedder.batch_extract(chunk, batch_size=embed_batch_size)

            if chunk_results:
                # Prepare data for NPZ
                embeddings = np.array([r['vector'] for r in chunk_results], dtype=np.float32)
                # Store metadata as JSON strings in a structured array
                metadatas = []
                ids = []
                for r in chunk_results:
                    meta = {
                        "lemma": r['lemma'],
                        "pos": r['pos'],
                        "sentence_id": r['sentence_id'],
                        "start_char": r['start_char'],
                        "end_char": r['end_char'],
                        "token": r.get('token', r['lemma'])
                    }
                    metadatas.append(json.dumps(meta))
                    ids.append(f"{collection_name}_{r['lemma']}_{r['pos']}_{r['sentence_id']}_{r['start_char']}")

                # Write NPZ file (compressed for space efficiency)
                npz_path = os.path.join(collection_staging_dir, f"chunk_{chunk_idx:06d}.npz")
                np.savez_compressed(
                    npz_path,
                    embeddings=embeddings,
                    metadatas=np.array(metadatas, dtype=object),
                    ids=np.array(ids, dtype=object)
                )
                total_written += len(chunk_results)
                chunk_idx += 1

            pbar.update(len(chunk))

    # Write manifest file with metadata about this collection
    manifest = {
        "collection_name": collection_name,
        "num_chunks": chunk_idx,
        "total_embeddings": total_written,
        "db_path": db_path,
        "embedding_dim": embedder.embedding_dim if hasattr(embedder, 'embedding_dim') else None
    }
    manifest_path = os.path.join(collection_staging_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"  Wrote {total_written} embeddings to {chunk_idx} NPZ files in {collection_staging_dir}")
    return total_written


def bulk_import_from_staging(staging_dir: str, vector_store: VectorStore,
                             model_name: str, tqdm_class: Type[tqdm_std] = tqdm,
                             delete_after_import: bool = False) -> Dict[str, int]:
    """
    Bulk import all staged NPZ files into ChromaDB.
    This builds the HNSW index once instead of incrementally.

    Args:
        staging_dir: Directory containing collection subdirectories with NPZ files
        vector_store: VectorStore instance
        model_name: Model name to store in collection metadata
        tqdm_class: Progress bar class
        delete_after_import: If True, delete NPZ files after successful import

    Returns:
        Dict mapping collection names to number of embeddings imported
    """
    results = {}

    # Find all collection directories
    if not os.path.exists(staging_dir):
        print(f"Staging directory not found: {staging_dir}")
        return results

    collection_dirs = [d for d in os.listdir(staging_dir)
                       if os.path.isdir(os.path.join(staging_dir, d))]

    if not collection_dirs:
        print(f"No collections found in {staging_dir}")
        return results

    print(f"Found {len(collection_dirs)} collections to import")

    for collection_name in collection_dirs:
        collection_dir = os.path.join(staging_dir, collection_name)
        manifest_path = os.path.join(collection_dir, "manifest.json")

        # Read manifest if it exists
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            total_expected = manifest.get('total_embeddings', 0)
        else:
            total_expected = None

        # Find all NPZ files
        npz_files = sorted(globlib.glob(os.path.join(collection_dir, "chunk_*.npz")))
        if not npz_files:
            print(f"  No NPZ files found in {collection_dir}, skipping")
            continue

        print(f"\n--- Importing {collection_name} ({len(npz_files)} files) ---")

        # Ensure collection exists with metadata
        vector_store.get_or_create_collection(collection_name, metadata={"model_name": model_name})

        # Accumulate all data first, then do single bulk insert
        all_embeddings = []
        all_metadatas = []
        all_ids = []

        for npz_path in tqdm_class(npz_files, desc=f"Loading {collection_name}"):
            data = np.load(npz_path, allow_pickle=True)
            all_embeddings.extend(data['embeddings'].tolist())
            all_metadatas.extend([json.loads(m) for m in data['metadatas']])
            all_ids.extend(data['ids'].tolist())

        total_loaded = len(all_embeddings)
        print(f"  Loaded {total_loaded} embeddings, inserting into ChromaDB...")

        # Single bulk insert - ChromaDB will batch internally if needed
        if all_embeddings:
            vector_store.add_embeddings(
                collection_name=collection_name,
                embeddings=[np.array(e) for e in all_embeddings],
                metadatas=all_metadatas,
                ids=all_ids,
                max_batch_size=10000  # Larger batches for bulk import
            )

        results[collection_name] = total_loaded
        print(f"  Successfully imported {total_loaded} embeddings to {collection_name}")

        # Optionally clean up NPZ files
        if delete_after_import:
            for npz_path in npz_files:
                os.remove(npz_path)
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
            try:
                os.rmdir(collection_dir)
            except OSError:
                pass  # Directory not empty or other issue
            print(f"  Cleaned up staging files for {collection_name}")

    return results


def get_staged_collections(staging_dir: str) -> List[Dict[str, Any]]:
    """
    List all staged collections and their status.
    Useful for checking what's ready to import or resuming interrupted work.
    """
    collections = []

    if not os.path.exists(staging_dir):
        return collections

    for name in os.listdir(staging_dir):
        dir_path = os.path.join(staging_dir, name)
        if not os.path.isdir(dir_path):
            continue

        manifest_path = os.path.join(dir_path, "manifest.json")
        npz_files = globlib.glob(os.path.join(dir_path, "chunk_*.npz"))

        info = {
            "name": name,
            "num_files": len(npz_files),
            "path": dir_path
        }

        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            info.update(manifest)

        collections.append(info)

    return collections


def get_collection_name(project_id: str, period: str, model_name: str) -> str:
    """
    Generates a consistent collection name.

    Args:
        project_id: 4-digit project identifier
        period: "t1" or "t2"
        model_name: HuggingFace model name

    Returns:
        Collection name in format: embeddings_{project_id}_{period}_{safe_model}
    """
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_{project_id}_{period}_{safe_model}"

def run_batch_generation(
    project_id: str,
    db_path_t1="data/corpus_t1.db",
    db_path_t2="data/corpus_t2.db",
    output_dir_t1="data/embeddings/t1",
    output_dir_t2="data/embeddings/t2",
    model_name="bert-base-uncased",
    min_freq=25,
    max_samples=200,
    additional_words: List[str] = None,
    db_path_1800=None,
    db_path_1900=None,
    output_dir_1800=None,
    output_dir_1900=None,
    reset_collections: bool = False,
    pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'),
    test_mode: bool = False,
    test_num_words: int = 25,
    test_samples_per_word: int = 50,
    embed_batch_size: int = None,
    tqdm_class: Type[tqdm_std] = tqdm,
    pooling_strategy: str = "mean",
    layers: List[int] = None,
    layer_op: str = "mean",
    staged: bool = False,
    staging_dir: str = None,
    import_only: bool = False,
    delete_staging_after_import: bool = False,
):
    """
    Entry point for the batch analysis/generation workflow.
    Processes frequent content words (Noun, Verb, Adj, Adv) by default.

    Args:
        test_mode: If True, uses limited words/samples for quick testing.
        test_num_words: Number of shared nouns to use in test mode (default 25).
        test_samples_per_word: Embeddings per word per period in test mode (default 50).
        embed_batch_size: Batch size for embedding extraction. If None, auto-detects based on GPU VRAM.
        tqdm_class: Progress bar class to use (tqdm for CLI, stqdm for Streamlit).
        pooling_strategy: How to pool subword tokens. Options:
            - 'mean': Mean of all subword tokens (default)
            - 'first': First subword token only
            - 'lemma_aligned': Only pool subwords matching lemma's tokenization length
            - 'weighted': Position-weighted pooling
            - 'lemma_replacement': Replace target with lemma before embedding (TokLem)
        layers: List of transformer layer indices to use (e.g., [-1] for last layer,
            [-4,-3,-2,-1] for last 4 layers). Default is [-1].
        layer_op: How to combine multiple layers: 'mean', 'median', 'sum', or 'concat'.
            Default is 'mean'. Note: 'concat' multiplies output dimension by len(layers).
        staged: If True, write embeddings to NPZ files first, then bulk import to ChromaDB.
            This is significantly faster for large batches due to reduced I/O overhead.
        staging_dir: Directory for NPZ staging files. Default: data/embedding_staging
        import_only: If True, skip embedding generation and only import existing staged files.
        delete_staging_after_import: If True, delete NPZ files after successful import.
    """

    # Handle legacy args
    if db_path_1800: db_path_t1 = db_path_1800
    if db_path_1900: db_path_t2 = db_path_1900

    # Set default staging directory
    if staging_dir is None:
        staging_dir = DEFAULT_STAGING_DIR

    # Test mode overrides
    if test_mode:
        print(f"=== TEST MODE: {test_num_words} shared nouns, {test_samples_per_word} samples/word ===")
        # Keep high frequency for test mode if not specified, but respect passed min_freq
        # min_freq = 50  <- Removed hardcoded value
        max_samples = test_samples_per_word
        pos_filter = 'NOUN'  # Only nouns in test mode

    start_time = time.time()
    import sys

    coll_t1 = get_collection_name(project_id, "t1", model_name)
    coll_t2 = get_collection_name(project_id, "t2", model_name)

    # Import-only mode: just load existing staged files into ChromaDB
    if import_only:
        print(f"=== IMPORT-ONLY MODE: Loading staged embeddings from {staging_dir} ===")
        sys.stdout.flush()

        # Check what's available
        staged = get_staged_collections(staging_dir)
        if not staged:
            print(f"No staged collections found in {staging_dir}")
            return

        print(f"Found {len(staged)} staged collections:")
        for s in staged:
            print(f"  - {s['name']}: {s.get('total_embeddings', '?')} embeddings in {s['num_files']} files")

        vector_store = VectorStore(persistence_path="data/chroma_db")
        results = bulk_import_from_staging(
            staging_dir=staging_dir,
            vector_store=vector_store,
            model_name=model_name,
            tqdm_class=tqdm_class,
            delete_after_import=delete_staging_after_import
        )

        elapsed_time = time.time() - start_time
        total_imported = sum(results.values())
        print(f"\nImport complete! Loaded {total_imported} embeddings in {elapsed_time/60:.2f} minutes.")
        return

    # Normal mode: generate embeddings
    print(f"Initializing ChromaDB Vector Store with target POS: {pos_filter}...")
    sys.stdout.flush()
    vector_store = VectorStore(persistence_path="data/chroma_db")

    # Read spaCy model from database metadata (for neighbor filtering)
    spacy_model = get_spacy_model_from_db(db_path_t1)
    if spacy_model:
        print(f"Using spaCy model from corpus metadata: {spacy_model}")
    else:
        print("No spaCy model found in corpus metadata, using defaults.")
    sys.stdout.flush()

    print(f"Loading embedding model '{model_name}'... (this may take a while for large models)")
    sys.stdout.flush()
    embedder = BertEmbedder(
        model_name=model_name,
        filter_model=spacy_model,
        pooling_strategy=pooling_strategy,
        layers=layers,
        layer_op=layer_op
    )
    print("Embedding model loaded successfully.")
    sys.stdout.flush()

    if reset_collections:
        print("--- Resetting Collections ---")
        vector_store.delete_collection(coll_t1)
        vector_store.delete_collection(coll_t2)
        # Also clear staging directory if using staged mode
        if staged:
            for coll in [coll_t1, coll_t2]:
                coll_staging = os.path.join(staging_dir, coll)
                if os.path.exists(coll_staging):
                    import shutil
                    shutil.rmtree(coll_staging)
                    print(f"  Cleared staging: {coll_staging}")

    if not staged:
        # Ensure collections are created with metadata (stores original model name)
        vector_store.get_or_create_collection(coll_t1, metadata={"model_name": model_name})
        vector_store.get_or_create_collection(coll_t2, metadata={"model_name": model_name})

    if test_mode:
        # Get shared words for test mode (same words in both periods)
        shared_words = get_shared_words(db_path_t1, db_path_t2, min_freq=min_freq, pos_filter=pos_filter)
        if len(shared_words) > test_num_words:
            import random
            shared_words = random.sample(shared_words, test_num_words)
        print(f"Test mode: Selected {len(shared_words)} shared nouns: {shared_words[:10]}...")
        words_t1 = shared_words
        words_t2 = shared_words
    else:
        # --- Period 1 ---
        words_t1 = get_frequent_words(db_path_t1, min_freq=min_freq, pos_filter=pos_filter)
        if additional_words:
            words_t1 = sorted(list(set(words_t1 + additional_words)))

        # --- Period 2 ---
        words_t2 = get_frequent_words(db_path_t2, min_freq=min_freq, pos_filter=pos_filter)
        if additional_words:
            words_t2 = sorted(list(set(words_t2 + additional_words)))

    if staged:
        # Staged mode: write to NPZ files first
        print(f"\n=== STAGED MODE: Writing embeddings to {staging_dir} ===")
        os.makedirs(staging_dir, exist_ok=True)

        print(f"\n--- Processing T1 Corpus -> NPZ (Collection: {coll_t1}) ---")
        process_corpus_staged(db_path_t1, coll_t1, words_t1, embedder, staging_dir,
                              max_samples=max_samples, pos_filter=pos_filter,
                              embed_batch_size=embed_batch_size, tqdm_class=tqdm_class)

        print(f"\n--- Processing T2 Corpus -> NPZ (Collection: {coll_t2}) ---")
        process_corpus_staged(db_path_t2, coll_t2, words_t2, embedder, staging_dir,
                              max_samples=max_samples, pos_filter=pos_filter,
                              embed_batch_size=embed_batch_size, tqdm_class=tqdm_class)

        # Now bulk import to ChromaDB
        print(f"\n=== Bulk importing staged embeddings to ChromaDB ===")
        results = bulk_import_from_staging(
            staging_dir=staging_dir,
            vector_store=vector_store,
            model_name=model_name,
            tqdm_class=tqdm_class,
            delete_after_import=delete_staging_after_import
        )

        elapsed_time = time.time() - start_time
        total_embeddings = sum(results.values())
        print(f"\nAll done! Staged batch generation completed in {elapsed_time/60:.2f} minutes.")
        print(f"Total embeddings: {total_embeddings}")
    else:
        # Direct mode: write to ChromaDB immediately (original behavior)
        print(f"\n--- Processing T1 Corpus (Collection: {coll_t1}) ---")
        process_corpus(db_path_t1, coll_t1, words_t1, embedder, vector_store, max_samples=max_samples,
                       pos_filter=pos_filter, embed_batch_size=embed_batch_size, tqdm_class=tqdm_class)

        print(f"\n--- Processing T2 Corpus (Collection: {coll_t2}) ---")
        process_corpus(db_path_t2, coll_t2, words_t2, embedder, vector_store, max_samples=max_samples,
                       pos_filter=pos_filter, embed_batch_size=embed_batch_size, tqdm_class=tqdm_class)

        elapsed_time = time.time() - start_time
        print(f"\nAll done! Batch generation completed in {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Batch Semantic Change Analysis (Embeddings Generation).")
    parser.add_argument("--project", type=str, default="", help="4-digit project ID (uses active project if not specified)")
    parser.add_argument("--db-t1", type=str, default="data/corpus_t1.db", help="Path to Period 1 DB")
    parser.add_argument("--db-t2", type=str, default="data/corpus_t2.db", help="Path to Period 2 DB")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--min-freq", type=int, default=25, help="Minimum frequency for content words")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per word per period")
    parser.add_argument("--words", type=str, default="", help="Comma-separated list of additional words")
    parser.add_argument("--reset", action="store_true", help="Delete existing collections before processing")
    parser.add_argument("--pos", type=str, default="NOUN,VERB,ADJ,ADV", help="Comma-separated POS tags to include")
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size (auto-detects based on GPU VRAM if not specified)")
    parser.add_argument("--pooling", type=str, default="mean",
                       choices=["mean", "first", "lemma_aligned", "weighted", "lemma_replacement"],
                       help="Subword pooling strategy: mean (default), first, lemma_aligned, weighted, lemma_replacement (TokLem)")
    parser.add_argument("--layers", type=str, default="-1",
                       help="Comma-separated layer indices (e.g., '-1' for last layer, '-4,-3,-2,-1' for last 4 layers)")
    parser.add_argument("--layer-op", type=str, default="mean",
                       choices=["mean", "median", "sum", "concat"],
                       help="How to combine multiple layers: mean (default), median, sum, or concat")
    parser.add_argument("--staged", action="store_true",
                       help="Use staged mode: write embeddings to NPZ files first, then bulk import to ChromaDB. "
                            "Significantly faster for large batches due to reduced I/O overhead.")
    parser.add_argument("--staging-dir", type=str, default=None,
                       help=f"Directory for NPZ staging files (default: {DEFAULT_STAGING_DIR})")
    parser.add_argument("--import-only", action="store_true",
                       help="Skip embedding generation, only import existing staged NPZ files to ChromaDB")
    parser.add_argument("--delete-staging", action="store_true",
                       help="Delete NPZ staging files after successful import to ChromaDB")

    args = parser.parse_args()

    # Get or create project
    from semantic_change.project_manager import ProjectManager
    pm = ProjectManager()
    if args.project:
        project_id = args.project
    else:
        project_id = pm.ensure_default_project(db_t1=args.db_t1, db_t2=args.db_t2)
        print(f"Using project: {project_id}")

    user_words = [w.strip() for w in args.words.split(',')] if args.words else []
    user_pos = [p.strip().upper() for p in args.pos.split(',')] if args.pos else ('NOUN', 'VERB', 'ADJ', 'ADV')
    user_layers = [int(l.strip()) for l in args.layers.split(',')] if args.layers else [-1]

    run_batch_generation(
        project_id=project_id,
        db_path_t1=args.db_t1,
        db_path_t2=args.db_t2,
        model_name=args.model,
        min_freq=args.min_freq,
        max_samples=args.max_samples,
        additional_words=user_words,
        reset_collections=args.reset,
        pos_filter=user_pos,
        embed_batch_size=args.batch_size,
        pooling_strategy=args.pooling,
        layers=user_layers,
        layer_op=args.layer_op,
        staged=args.staged,
        staging_dir=args.staging_dir,
        import_only=args.import_only,
        delete_staging_after_import=args.delete_staging
    )