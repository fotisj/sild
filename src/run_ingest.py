from semantic_change.ingestor import Ingestor
from semantic_change.reporting import generate_comparison_report
from semantic_change.corpus import Corpus
import time

def main():
    start_time = time.time()
    print("--- Starting Full Corpus Ingestion ---")
    # Use high-quality model for ingestion
    ingestor = Ingestor(model="en_core_web_lg")
    
    # Process 1800 (limited for testing)
    print("Processing 1800 corpus...")
    db_path_1800 = "data/corpus_t1.db"
    ingestor.preprocess_corpus("data_gutenberg/1800", db_path_1800, max_files=30)
    
    # Process 1900 (limited for testing)
    print("Processing 1900 corpus...")
    db_path_1900 = "data/corpus_t2.db"
    ingestor.preprocess_corpus("data_gutenberg/1900", db_path_1900, max_files=30)
    
    # Load Corpora
    corpus_1800 = Corpus("1800", "data_gutenberg/1800", db_path_1800)
    corpus_1900 = Corpus("1900", "data_gutenberg/1900", db_path_1900)
    
    # Generate Report
    generate_comparison_report(corpus_1800, corpus_1900, top_n=50)

    elapsed_time = time.time() - start_time
    print(f"--- Ingestion Complete in {elapsed_time/60:.2f} minutes ---")

if __name__ == "__main__":
    main()
