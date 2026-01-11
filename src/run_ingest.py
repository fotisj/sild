"""
Corpus Ingestion Module

Provides functions for ingesting raw text corpora into SQLite databases.
Can be used via CLI or imported by the GUI.
"""
from semantic_change.ingestor import Ingestor
from semantic_change.reporting import generate_comparison_report
from semantic_change.corpus import Corpus
import time
import argparse
import os
from typing import Callable, Optional


def run_ingestion(
    input_t1: str,
    input_t2: str,
    db_t1: str,
    db_t2: str,
    label_t1: str = "t1",
    label_t2: str = "t2",
    spacy_model: str = "en_core_web_lg",
    file_encoding: str = "utf-8",
    max_files: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    generate_report: bool = True,
    report_top_n: int = 50,
) -> None:
    """
    Run the full corpus ingestion pipeline.

    Args:
        input_t1: Input directory for Period 1 text files
        input_t2: Input directory for Period 2 text files
        db_t1: Output SQLite database path for Period 1
        db_t2: Output SQLite database path for Period 2
        label_t1: Label for Period 1 (used in reports)
        label_t2: Label for Period 2 (used in reports)
        spacy_model: SpaCy model for tokenization/lemmatization
        file_encoding: Text file encoding (e.g., 'utf-8', 'cp1252', 'latin-1')
        max_files: Limit to N random files per period (for testing). None = all files.
        progress_callback: Optional callback(current, total, description) for progress updates
        generate_report: Whether to generate a comparison report after ingestion
        report_top_n: Number of top words to include in the report
    """
    print(f"--- Processing {label_t1} from {input_t1} ---")
    ingestor = Ingestor(model=spacy_model, encoding=file_encoding)
    ingestor.preprocess_corpus(input_t1, db_t1, max_files=max_files, progress_callback=progress_callback)

    print(f"--- Processing {label_t2} from {input_t2} ---")
    ingestor.preprocess_corpus(input_t2, db_t2, max_files=max_files, progress_callback=progress_callback)

    if generate_report:
        print("--- Generating Comparison Report ---")
        corpus_t1 = Corpus(label_t1, input_t1, db_t1)
        corpus_t2 = Corpus(label_t2, input_t2, db_t2)
        generate_comparison_report(corpus_t1, corpus_t2, top_n=report_top_n)


def delete_databases(db_t1: str, db_t2: str) -> tuple[bool, str]:
    """
    Delete the corpus databases.

    Returns:
        Tuple of (success, message)
    """
    try:
        deleted = []
        if os.path.exists(db_t1):
            os.remove(db_t1)
            deleted.append(db_t1)
        if os.path.exists(db_t2):
            os.remove(db_t2)
            deleted.append(db_t2)

        if deleted:
            return True, f"Deleted: {', '.join(deleted)}"
        return True, "No databases to delete"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser(description="Run Full Corpus Ingestion.")
    parser.add_argument("--input-t1", type=str, default="data_source/t1", help="Input directory for Period 1")
    parser.add_argument("--input-t2", type=str, default="data_source/t2", help="Input directory for Period 2")
    parser.add_argument("--db-t1", type=str, default="data/corpus_t1.db", help="Output DB path for Period 1")
    parser.add_argument("--db-t2", type=str, default="data/corpus_t2.db", help="Output DB path for Period 2")
    parser.add_argument("--label-t1", type=str, default="t1", help="Label for Period 1 (for report)")
    parser.add_argument("--label-t2", type=str, default="t2", help="Label for Period 2 (for report)")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_lg", help="SpaCy model to use")
    parser.add_argument("--encoding", type=str, default="utf-8", help="Text file encoding (utf-8, cp1252, latin-1)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit to N random files per period (for testing)")

    args = parser.parse_args()

    start_time = time.time()
    print("--- Starting Full Corpus Ingestion ---")

    run_ingestion(
        input_t1=args.input_t1,
        input_t2=args.input_t2,
        db_t1=args.db_t1,
        db_t2=args.db_t2,
        label_t1=args.label_t1,
        label_t2=args.label_t2,
        spacy_model=args.spacy_model,
        file_encoding=args.encoding,
        max_files=args.max_files,
    )

    elapsed_time = time.time() - start_time
    print(f"--- Ingestion Complete in {elapsed_time/60:.2f} minutes ---")


if __name__ == "__main__":
    main()
