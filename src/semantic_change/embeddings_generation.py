import sqlite3
import torch
import numpy as np
import os
import time
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Dict, Tuple, Set, Any, Union
from .embedding import BertEmbedder
from .vector_store import VectorStore
from .corpus import get_spacy_model_from_db
from tqdm import tqdm

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

def collect_sentences_for_words(db_path: str, target_words: List[str], max_samples: int = 500, pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'), progress_callback=None) -> List[Dict[str, Any]]:
    """
    Retrieves sentences containing the target words from the SQLite database.
    Optimized to fetch in batches to avoid N+1 query problem.
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
    iterator = range(0, len(target_words), word_chunk_size)
    if not progress_callback:
        iterator = tqdm(iterator, desc="Preparing word samples")
        
    total_words = len(target_words)
    
    for i in iterator:
        chunk_words = target_words[i : i + word_chunk_size]
        placeholders = ','.join('?' for _ in chunk_words)
        
        # We need a way to limit per word in SQL, or we fetch all and limit in Python.
        # Fetching all matches for frequently occurring words might be heavy.
        # Window functions would be ideal (ROW_NUMBER() OVER (PARTITION BY lemma)) but require SQLite 3.25+
        # Let's assume SQLite 3.25+ is available or fallback to a slightly heavier fetch.
        
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
            # This is slower but safe
            query = f"SELECT sentence_id, start_char, text, pos, lemma FROM tokens WHERE lemma IN ({placeholders}) AND {pos_condition}"
            params = chunk_words + pos_params
            rows = cursor.execute(query, params).fetchall()
            
            # Group by lemma to limit
            counts = defaultdict(int)
            import random
            random.shuffle(rows) # Simple randomization
            
            for sent_id, start_char, token_text, token_pos, lemma in rows:
                if counts[lemma] < max_samples:
                    unique_sentence_ids.add(sent_id)
                    end_char = start_char + len(token_text)
                    sentence_tasks[sent_id].append((lemma, token_pos, start_char, end_char))
                    counts[lemma] += 1

        if progress_callback:
            progress_callback(min(i + word_chunk_size, total_words), total_words, "Preparing word samples")

    sent_ids_list = list(unique_sentence_ids)
    chunk_size = 999
    
    # 2. Fetching texts
    desc = "Fetching texts"
    total_chunks = len(range(0, len(sent_ids_list), chunk_size))
    
    batch_items = [] 
    
    if progress_callback:
        progress_callback(0, total_chunks, desc)
        
    chunk_iterator = range(0, len(sent_ids_list), chunk_size)
    if not progress_callback:
        chunk_iterator = tqdm(chunk_iterator, desc=desc)
        
    for i, start_idx in enumerate(chunk_iterator):
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
                
        if progress_callback:
            progress_callback(i + 1, total_chunks, desc)
            
    conn.close()
    return batch_items

def process_corpus(db_path: str, collection_name: str, target_words: List[str],
                   embedder: BertEmbedder, vector_store: VectorStore, max_samples: int = 200,
                   pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'),
                   progress_callback=None):
    """
    Orchestrates the processing of a single corpus: collection, embedding extraction, and saving to ChromaDB.
    Uses async DB writes to overlap with GPU computation.
    """
    batch_items = collect_sentences_for_words(db_path, target_words, max_samples=max_samples, pos_filter=pos_filter, progress_callback=progress_callback)
    total_sents = len(batch_items)
    if total_sents == 0:
        print(f"No sentences found for {os.path.basename(db_path)}.")
        return

    # Increased chunk size to reduce ChromaDB write frequency (avoids compaction errors)
    chunk_size = 200
    # Larger batch size for embedding extraction
    embed_batch_size = 128
    desc = f"Processing {os.path.basename(db_path)} -> ChromaDB: {collection_name}"

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
                "end_char": r['end_char']
            } for r in chunk_results],
            ids=[f"{collection_name}_{r['lemma']}_{r['pos']}_{r['sentence_id']}_{r['start_char']}" for r in chunk_results]
        )

    # Use thread pool to overlap DB writes with GPU computation
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_future: Future = None

        if progress_callback:
            progress_callback(0, total_sents, desc)
            for i in range(0, total_sents, chunk_size):
                chunk = batch_items[i : i + chunk_size]
                chunk_results = embedder.batch_extract(chunk, batch_size=embed_batch_size)

                # Wait for previous write to complete before submitting new one
                if pending_future is not None:
                    pending_future.result()

                # Submit DB write asynchronously
                pending_future = executor.submit(save_chunk, chunk_results)
                progress_callback(min(i + chunk_size, total_sents), total_sents, desc)

            # Wait for final write
            if pending_future is not None:
                pending_future.result()
        else:
            with tqdm(total=total_sents, desc=desc) as pbar:
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
    progress_callback=None,
    db_path_1800=None,
    db_path_1900=None,
    output_dir_1800=None,
    output_dir_1900=None,
    reset_collections: bool = False,
    pos_filter: Union[str, List[str], Tuple[str, ...]] = ('NOUN', 'VERB', 'ADJ', 'ADV'),
    test_mode: bool = False,
    test_num_words: int = 25,
    test_samples_per_word: int = 50,
):
    """
    Entry point for the batch analysis/generation workflow.
    Processes frequent content words (Noun, Verb, Adj, Adv) by default.

    Args:
        test_mode: If True, uses limited words/samples for quick testing.
        test_num_words: Number of shared nouns to use in test mode (default 25).
        test_samples_per_word: Embeddings per word per period in test mode (default 50).
    """

    # Handle legacy args
    if db_path_1800: db_path_t1 = db_path_1800
    if db_path_1900: db_path_t2 = db_path_1900

    # Test mode overrides
    if test_mode:
        print(f"=== TEST MODE: {test_num_words} shared nouns, {test_samples_per_word} samples/word ===")
        # Keep high frequency for test mode if not specified, but respect passed min_freq
        # min_freq = 50  <- Removed hardcoded value
        max_samples = test_samples_per_word
        pos_filter = 'NOUN'  # Only nouns in test mode

    start_time = time.time()
    import sys

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
    embedder = BertEmbedder(model_name=model_name, filter_model=spacy_model)
    print("Embedding model loaded successfully.")
    sys.stdout.flush()

    coll_t1 = get_collection_name(project_id, "t1", model_name)
    coll_t2 = get_collection_name(project_id, "t2", model_name)

    if reset_collections:
        print("--- Resetting Collections ---")
        vector_store.delete_collection(coll_t1)
        vector_store.delete_collection(coll_t2)

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

    print(f"\n--- Processing T1 Corpus (Collection: {coll_t1}) ---")
    process_corpus(db_path_t1, coll_t1, words_t1, embedder, vector_store, max_samples=max_samples, pos_filter=pos_filter, progress_callback=progress_callback)

    print(f"\n--- Processing T2 Corpus (Collection: {coll_t2}) ---")
    process_corpus(db_path_t2, coll_t2, words_t2, embedder, vector_store, max_samples=max_samples, pos_filter=pos_filter, progress_callback=progress_callback)

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

    run_batch_generation(
        project_id=project_id,
        db_path_t1=args.db_t1,
        db_path_t2=args.db_t2,
        model_name=args.model,
        min_freq=args.min_freq,
        max_samples=args.max_samples,
        additional_words=user_words,
        reset_collections=args.reset,
        pos_filter=user_pos
    )