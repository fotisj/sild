import sqlite3
import torch
import numpy as np
import os
import time
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Any
from semantic_change.embedding import BertEmbedder
from semantic_change.vector_store import VectorStore
from tqdm import tqdm

def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def get_frequent_words(db_path: str, min_freq: int = 25, pos_filter: str = 'NOUN') -> List[str]:
    """
    Finds words in a specific database with frequency >= min_freq.
    """
    print(f"Finding nouns in {os.path.basename(db_path)} with frequency >= {min_freq}...")
    conn = get_db_connection(db_path)
    
    # Check if table exists
    try:
        query = f"SELECT lemma, count(*) as c FROM tokens WHERE pos='{pos_filter}' GROUP BY lemma HAVING c >= {min_freq}"
        rows = conn.execute(query).fetchall()
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

    # NLTK filtering removed to support multi-language/historical corpora.
    # We rely on spaCy's lemmatization and frequency thresholds.

    for w in words:
        if len(w) < 2: continue # Keep length check minimal
        # Alpha check might be too strict for some languages (e.g. hyphens)? 
        # But generally good for "words". Let's keep isalpha for now, or allow hyphens.
        if not w.replace('-', '').isalpha(): continue 
        if w in stop_words: continue
        
        clean_words.append(w)
        
    print(f"Found {len(clean_words)} valid words in {os.path.basename(db_path)}.")
    return sorted(list(clean_words))

def collect_sentences_for_words(db_path: str, target_words: List[str], max_samples: int = 500, pos_filter: str = 'NOUN', progress_callback=None) -> List[Dict[str, Any]]:
    """
    Retrieves sentences containing the target words from the SQLite database.
    """
    print(f"Collecting sentences from {db_path} (max_samples={max_samples}, pos={pos_filter})...")
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    sentence_tasks = defaultdict(list)
    unique_sentence_ids = set()
    
    # 1. Preparing word samples
    iterator = target_words
    total = len(target_words)
    desc = "Preparing word samples"
    
    # Helper to process rows
    def process_rows(rows, word):
        for sent_id, start_char, token_text in rows:
            unique_sentence_ids.add(sent_id)
            end_char = start_char + len(token_text)
            sentence_tasks[sent_id].append((word, start_char, end_char))

    if progress_callback:
        progress_callback(0, total, desc)
        for i, word in enumerate(iterator):
            query = "SELECT sentence_id, start_char, text FROM tokens WHERE lemma = ? AND pos = ? LIMIT ?"
            rows = cursor.execute(query, (word, pos_filter, max_samples)).fetchall()
            process_rows(rows, word)
            
            if i % 10 == 0:
                progress_callback(i + 1, total, desc)
        progress_callback(total, total, desc)
    else:
        for word in tqdm(target_words, desc=desc):
            query = "SELECT sentence_id, start_char, text FROM tokens WHERE lemma = ? AND pos = ? LIMIT ?"
            rows = cursor.execute(query, (word, pos_filter, max_samples)).fetchall()
            process_rows(rows, word)
            
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
                   embedder: BertEmbedder, vector_store: VectorStore, max_samples: int = 200, progress_callback=None):
    """
    Orchestrates the processing of a single corpus: collection, embedding extraction, and saving to ChromaDB.
    """
    batch_items = collect_sentences_for_words(db_path, target_words, max_samples=max_samples, progress_callback=progress_callback)
    total_sents = len(batch_items)
    if total_sents == 0:
        print(f"No sentences found for {os.path.basename(db_path)}.")
        return

    chunk_size = 500 
    desc = f"Processing {os.path.basename(db_path)} -> ChromaDB: {collection_name}"
    
    def save_chunk(chunk_results):
        if not chunk_results: return
        
        vector_store.add_embeddings(
            collection_name=collection_name,
            embeddings=[r['vector'] for r in chunk_results],
            metadatas=[{
                "lemma": r['lemma'],
                "sentence_id": r['sentence_id'],
                "start_char": r['start_char'],
                "end_char": r['end_char']
            } for r in chunk_results],
            ids=[f"{collection_name}_{r['lemma']}_{r['sentence_id']}_{r['start_char']}" for r in chunk_results]
        )

    if progress_callback:
        progress_callback(0, total_sents, desc)
        for i in range(0, total_sents, chunk_size):
            chunk = batch_items[i : i+chunk_size]
            chunk_results = embedder.batch_extract(chunk, batch_size=64)
            save_chunk(chunk_results)
            progress_callback(min(i + chunk_size, total_sents), total_sents, desc)
    else:
        with tqdm(total=total_sents, desc=desc) as pbar:
            for i in range(0, total_sents, chunk_size):
                chunk = batch_items[i : i+chunk_size]
                chunk_results = embedder.batch_extract(chunk, batch_size=64)
                save_chunk(chunk_results)
                pbar.update(len(chunk))

def get_collection_name(period_id: str, model_name: str) -> str:
    """Generates a consistent collection name based on internal period ID (t1/t2) and model."""
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_{period_id}_{safe_model}"

def run_batch_process(

    db_path_t1="data/corpus_t1.db",

    db_path_t2="data/corpus_t2.db",

    output_dir_t1="data/embeddings/t1", # Deprecated but kept for signature compatibility

    output_dir_t2="data/embeddings/t2", # Deprecated

    model_name="bert-base-uncased",

    min_freq=25,

    max_samples=200,

    additional_words: List[str] = None,

    progress_callback=None,

    # Compatibility with old calls using 1800/1900 args

    db_path_1800=None,

    db_path_1900=None,

    output_dir_1800=None,

    output_dir_1900=None,

    reset_collections: bool = False

):

    """

    Entry point for the batch analysis workflow.

    Processes frequent words for each corpus independently using t1/t2 naming.

    """

    # Handle legacy args

    if db_path_1800: db_path_t1 = db_path_1800

    if db_path_1900: db_path_t2 = db_path_1900



    start_time = time.time()

    

    print("Initializing ChromaDB Vector Store...")

    vector_store = VectorStore(persistence_path="data/chroma_db")

    

    embedder = BertEmbedder(model_name=model_name)

    

    coll_t1 = get_collection_name("t1", model_name)

    coll_t2 = get_collection_name("t2", model_name)



    if reset_collections:

        print("--- Resetting Collections ---")

        vector_store.delete_collection(coll_t1)

        vector_store.delete_collection(coll_t2)

    

    # --- Period 1 ---

    words_t1 = get_frequent_words(db_path_t1, min_freq=min_freq)

    if additional_words:

        words_t1 = sorted(list(set(words_t1 + additional_words)))

        

    print(f"\n--- Processing T1 Corpus (Collection: {coll_t1}) ---")

    process_corpus(db_path_t1, coll_t1, words_t1, embedder, vector_store, max_samples=max_samples, progress_callback=progress_callback)

    

    # --- Period 2 ---

    words_t2 = get_frequent_words(db_path_t2, min_freq=min_freq)

    if additional_words:

        words_t2 = sorted(list(set(words_t2 + additional_words)))



    print(f"\n--- Processing T2 Corpus (Collection: {coll_t2}) ---")

    process_corpus(db_path_t2, coll_t2, words_t2, embedder, vector_store, max_samples=max_samples, progress_callback=progress_callback)

    

    elapsed_time = time.time() - start_time

    print(f"\nAll done! Batch analysis completed in {elapsed_time/60:.2f} minutes.")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run Batch Semantic Change Analysis.")

    parser.add_argument("--db-t1", type=str, default="data/corpus_t1.db", help="Path to Period 1 DB")

    parser.add_argument("--db-t2", type=str, default="data/corpus_t2.db", help="Path to Period 2 DB")

    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")

    parser.add_argument("--min-freq", type=int, default=25, help="Minimum frequency for nouns")

    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per word per period")

    parser.add_argument("--words", type=str, default="", help="Comma-separated list of additional words")

    parser.add_argument("--reset", action="store_true", help="Delete existing collections before processing")

    

    args = parser.parse_args()

    

    user_words = [w.strip() for w in args.words.split(',')] if args.words else []

    

    run_batch_process(

        db_path_t1=args.db_t1, 

        db_path_t2=args.db_t2,

        model_name=args.model,

        min_freq=args.min_freq,

        max_samples=args.max_samples,

        additional_words=user_words,

        reset_collections=args.reset

    )
