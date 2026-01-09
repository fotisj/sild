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

def get_top_shared_words(db1: str, db2: str, min_freq: int = 25, additional_words: List[str] = None) -> List[str]:
    print(f"Finding all shared nouns with frequency >= {min_freq}...")
    conn1 = get_db_connection(db1)
    conn2 = get_db_connection(db2)
    
    # We remove the LIMIT and sort logic here, relying on frequency threshold
    query = f"SELECT lemma, count(*) as c FROM tokens WHERE pos='NOUN' GROUP BY lemma HAVING c >= {min_freq}"
    
    top1 = {row[0] for row in conn1.execute(query).fetchall()}
    top2 = {row[0] for row in conn2.execute(query).fetchall()}
    
    conn1.close()
    conn2.close()
    
    shared = list(top1.intersection(top2))
    
    # Filter out very short or non-alpha artifacts
    # Also filter stopwords and likely foreign/garbage tokens
    print("Filtering vocabulary...")
    try:
        import nltk
        try:
            english_vocab = set(nltk.corpus.words.words())
        except LookupError:
            print("Downloading nltk words...")
            nltk.download('words')
            english_vocab = set(nltk.corpus.words.words())
    except ImportError:
        print("NLTK not found. Vocabulary filtering will be basic.")
        english_vocab = None

    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        stop_words = nlp.Defaults.stop_words
    except:
        stop_words = set()

    clean_shared = []
    for w in shared:
        if len(w) < 3: continue
        if not w.isalpha(): continue
        if w in stop_words: continue
        
        # Heuristic: If it's in NLTK words, keep it. 
        # If not, check if it's common enough or just garbage? 
        # For now, if we have NLTK, strict filter.
        if english_vocab and w not in english_vocab:
            # Maybe it's a plural form not in dict?
            # spaCy lemmas should be singular.
            # Allow some exceptions if needed, but for now strict is safer for "mit"/"des".
            continue
            
        clean_shared.append(w)
        
    print(f"Found {len(clean_shared)} valid shared nouns (filtered from {len(shared)}) with freq >= {min_freq}.")
    
    final_words = clean_shared
    
    # Add user-defined words if provided
    if additional_words:
        print(f"Adding {len(additional_words)} user-defined words.")
        existing_set = set(final_words)
        for w in additional_words:
            w_clean = w.strip().lower()
            if w_clean and w_clean not in existing_set:
                final_words.append(w_clean)
                existing_set.add(w_clean)
                
    return final_words

def collect_sentences_for_words(db_path: str, target_words: List[str], max_samples: int = 500, pos_filter: str = 'NOUN', progress_callback=None) -> List[Dict[str, Any]]:
    """
    Retrieves sentences containing the target words from the SQLite database.
    
    Args:
        db_path: Path to the SQLite database.
        target_words: List of word lemmas to search for.
        max_samples: Maximum number of samples to retrieve per word.
        pos_filter: POS tag to filter by (default 'NOUN').
        progress_callback: Optional function(current, total, desc) for progress updates.
        
    Returns:
        List of dictionaries: {'id': sent_id, 'text': text, 'targets': [(lemma, start, end), ...]}.
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
            # We now trust start_char as relative offset
            end_char = start_char + len(token_text)
            sentence_tasks[sent_id].append((word, start_char, end_char))

    if progress_callback:
        progress_callback(0, total, desc)
        for i, word in enumerate(iterator):
            # STRICT filtering by POS and Lemma
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
    
    batch_items = [] # The result list
    
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
            # sentence_tasks[sid] is a list of (lemma, start, end)
            # Since we have precise offsets, we just pass them through
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
    chunk_size = 500 # Slightly smaller chunk for DB writes
    
    desc = f"Processing {os.path.basename(db_path)} -> ChromaDB: {collection_name}"
    
    # Helper to add to DB
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

def get_collection_name(period: str, model_name: str) -> str:
    """Generates a consistent collection name based on period and model."""
    safe_model = model_name.replace("/", "_").replace("-", "_")
    return f"embeddings_{period}_{safe_model}"

def run_batch_process(
    db_path_1800="data/corpus_t1.db",
    db_path_1900="data/corpus_t2.db",
    output_dir_1800="data/embeddings/1800", # Deprecated but kept for signature compatibility
    output_dir_1900="data/embeddings/1900", # Deprecated
    model_name="bert-base-uncased",
    min_freq=25,
    max_samples=200,
    additional_words: List[str] = None,
    progress_callback=None
):
    """
    Entry point for the batch analysis workflow.
    Identifies shared words and processes both corpora.
    """
    start_time = time.time()
    
    # Initialize Vector Store
    print("Initializing ChromaDB Vector Store...")
    vector_store = VectorStore(persistence_path="data/chroma_db")
    
    shared_words = get_top_shared_words(db_path_1800, db_path_1900, min_freq=min_freq, additional_words=additional_words)
    
    # We load the embedder here. 
    embedder = BertEmbedder(model_name=model_name)
    
    coll_1800 = get_collection_name("1800", model_name)
    coll_1900 = get_collection_name("1900", model_name)
    
    print(f"\n--- Processing 1800 Corpus (Collection: {coll_1800}) ---")
    process_corpus(db_path_1800, coll_1800, shared_words, embedder, vector_store, max_samples=max_samples, progress_callback=progress_callback)
    
    print(f"\n--- Processing 1900 Corpus (Collection: {coll_1900}) ---")
    process_corpus(db_path_1900, coll_1900, shared_words, embedder, vector_store, max_samples=max_samples, progress_callback=progress_callback)
    
    elapsed_time = time.time() - start_time
    print(f"\nAll done! Batch analysis completed in {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Batch Semantic Change Analysis.")
    parser.add_argument("--db1800", type=str, default="data/corpus_t1.db", help="Path to 1800 DB")
    parser.add_argument("--db1900", type=str, default="data/corpus_t2.db", help="Path to 1900 DB")
    parser.add_argument("--model", type=str, default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--min-freq", type=int, default=25, help="Minimum frequency for shared nouns")
    parser.add_argument("--max-samples", type=int, default=200, help="Max samples per word per period")
    parser.add_argument("--words", type=str, default="", help="Comma-separated list of additional words to process")
    
    args = parser.parse_args()
    
    user_words = [w.strip() for w in args.words.split(',')] if args.words else []
    
    run_batch_process(
        db_path_1800=args.db1800, 
        db_path_1900=args.db1900,
        model_name=args.model,
        min_freq=args.min_freq,
        max_samples=50, # Reduced to 50 for speed
        additional_words=user_words
    )
