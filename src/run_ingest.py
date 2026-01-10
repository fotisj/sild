from semantic_change.ingestor import Ingestor
from semantic_change.reporting import generate_comparison_report
from semantic_change.corpus import Corpus
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run Full Corpus Ingestion.")
    parser.add_argument("--input-t1", type=str, default="data_gutenberg/1800", help="Input directory for Period 1")
    parser.add_argument("--input-t2", type=str, default="data_gutenberg/1900", help="Input directory for Period 2")
    parser.add_argument("--db-t1", type=str, default="data/corpus_t1.db", help="Output DB path for Period 1")
    parser.add_argument("--db-t2", type=str, default="data/corpus_t2.db", help="Output DB path for Period 2")
    parser.add_argument("--label-t1", type=str, default="1800", help="Label for Period 1 (for report)")
    parser.add_argument("--label-t2", type=str, default="1900", help="Label for Period 2 (for report)")
    
    args = parser.parse_args()

    start_time = time.time()
    print("--- Starting Full Corpus Ingestion ---")
    
    # Use high-quality model for ingestion
    ingestor = Ingestor(model="en_core_web_lg")
    
    # Process T1
    print(f"Processing {args.label_t1} corpus from {args.input_t1}...")
    ingestor.preprocess_corpus(args.input_t1, args.db_t1, max_files=None)
    
    # Process T2
    print(f"Processing {args.label_t2} corpus from {args.input_t2}...")
    ingestor.preprocess_corpus(args.input_t2, args.db_t2, max_files=None)
    
    # Load Corpora for Report
    corpus_t1 = Corpus(args.label_t1, args.input_t1, args.db_t1)
    corpus_t2 = Corpus(args.label_t2, args.input_t2, args.db_t2)
    
    # Generate Report
    print("Generating Comparison Report...")
    generate_comparison_report(corpus_t1, corpus_t2, top_n=50)

    elapsed_time = time.time() - start_time
    print(f"--- Ingestion Complete in {elapsed_time/60:.2f} minutes ---")

if __name__ == "__main__":
    main()