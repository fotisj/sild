import argparse
import pandas as pd
import numpy as np
import time
from scipy.spatial.distance import cosine
from semantic_change.vector_store import VectorStore
from semantic_change.embeddings_generation import get_shared_words
from tqdm import tqdm

def compute_centroid_shift(
    db1800="data/corpus_t1.db",
    db1900="data/corpus_t2.db",
    min_freq=25,
    output_file="output/semantic_change_ranking.csv",
    progress_callback=None,
    model_name="bert-base-uncased"
):
    print(f"--- Starting Semantic Change Ranking (Freq >= {min_freq}, Model: {model_name}) ---")
    
    # Collection naming
    safe_model = model_name.replace("/", "_").replace("-", "_")
    coll_1800 = f"embeddings_t1_{safe_model}"
    coll_1900 = f"embeddings_t2_{safe_model}"
    
    # 1. Identify words to analyze
    target_words = get_shared_words(db1800, db1900, min_freq=min_freq)
    total_words = len(target_words)
    
    print(f"Targeting {total_words} words for ranking.")
    
    # 2. Initialize Vector Store
    v_store = VectorStore(persistence_path="data/chroma_db")
    
    results = []
    
    # 3. Compute Shifts
    print(f"Computing distances using collections: {coll_1800} / {coll_1900}...")
    desc = "Computing centroids"
    
    if progress_callback:
        progress_callback(0, total_words, desc)
        
    iterator = enumerate(target_words)
    if not progress_callback:
        iterator = tqdm(target_words)
        
    for i, word in (enumerate(target_words) if progress_callback else enumerate(tqdm(target_words))):
        try:
            # Fetch 1800 (t1)
            # Use lemma only (no POS check)
            data_1800 = v_store.get_by_metadata(coll_1800, where={"lemma": word}, limit=2000)
            vecs_1800 = data_1800['embeddings']
            
            # Fetch 1900 (t2)
            data_1900 = v_store.get_by_metadata(coll_1900, where={"lemma": word}, limit=2000)
            vecs_1900 = data_1900['embeddings']
            
            count_1800 = len(vecs_1800) if vecs_1800 is not None else 0
            count_1900 = len(vecs_1900) if vecs_1900 is not None else 0
            
            # We need enough data to be significant. 
            if count_1800 < 5 or count_1900 < 5:
                if progress_callback and i % 10 == 0:
                    progress_callback(i + 1, total_words, desc)
                continue
                
            # Compute Centroids
            centroid_1800 = np.mean(vecs_1800, axis=0)
            centroid_1900 = np.mean(vecs_1900, axis=0)
            
            # Cosine Distance
            dist = cosine(centroid_1800, centroid_1900)
            
            results.append({
                "word": word,
                "distance": dist,
                "count_1800": count_1800,
                "count_1900": count_1900
            })
            
        except Exception as e:
            print(f"Error processing {word}: {e}")
            pass
        
        if progress_callback and i % 5 == 0:
            progress_callback(i + 1, total_words, desc)

    if progress_callback:
        progress_callback(total_words, total_words, "Ranking complete")

    # 4. Save Results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values(by="distance", ascending=False)
        
        df.to_csv(output_file, index=False)
        print(f"\nRanking complete. Saved top results to {output_file}")
        print(df.head(10))
    else:
        print("No results computed. Ensure the Vector DB is populated via 'run_batch_analysis.py'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-freq", type=int, default=25)
    parser.add_argument("--output", type=str, default="output/semantic_change_ranking.csv")
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    args = parser.parse_args()
    
    compute_centroid_shift(min_freq=args.min_freq, output_file=args.output, model_name=args.model)
