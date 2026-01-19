import os
import sqlite3
import spacy
from typing import List, Dict, Any, Tuple, Type
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from tqdm.std import tqdm as tqdm_std

# Proactively import spaCy transformer packages if installed
# This registers their factories with spaCy before any model load
try:
    import spacy_curated_transformers
except ImportError:
    pass

try:
    import spacy_transformers
except ImportError:
    pass


class Ingestor:
    def __init__(self, model: str = "en_core_web_lg", encoding: str = "utf-8"):
        self.encoding = encoding
        self.model_name = model

        # Check for GPU availability
        if spacy.prefer_gpu():
            print("GPU usage enabled for spaCy.")
        else:
            print("GPU not available or not supported by spaCy installation. Using CPU.")

        # Load spaCy model (download if needed for non-transformer models)
        print(f"Loading spaCy model: {model}")
        try:
            self.nlp = spacy.load(model, disable=["parser", "ner"])
        except OSError:
            print(f"Model '{model}' not found. Attempting to download...")
            self._download_model(model)
            self.nlp = spacy.load(model, disable=["parser", "ner"])

        self.nlp.add_pipe('sentencizer')
        self.nlp.max_length = 6000000  # Increased limit

    def _download_model(self, model: str) -> None:
        """Download a spaCy model if not installed."""
        import subprocess
        import sys

        print(f"Downloading spaCy model: {model}")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Download output: {result.stdout}")
            print(f"Download errors: {result.stderr}")
            raise OSError(f"Failed to download spaCy model '{model}'. Check the model name is valid.")
        print(f"Successfully downloaded: {model}")

    def _init_db(self, db_path: Path):
        """Initialize the SQLite database with the required schema."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency/performance
        cursor.execute("PRAGMA journal_mode=WAL;")
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                filename TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER,
                text TEXT,
                file_offset_start INTEGER,
                FOREIGN KEY(file_id) REFERENCES files(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id INTEGER,
                text TEXT,
                lemma TEXT,
                pos TEXT,
                start_char INTEGER,
                FOREIGN KEY(sentence_id) REFERENCES sentences(id)
            )
        """)
        
        # Create indices for fast lookups
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lemma ON tokens(lemma)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pos ON tokens(pos)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sent_file ON sentences(file_id)")

        # Store metadata about the ingestion
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("spacy_model", self.model_name)
        )
        cursor.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("encoding", self.encoding)
        )

        conn.commit()
        return conn

    def process_file_to_db(self, file_path: str, conn: sqlite3.Connection):
        """
        Processes a single text file and inserts data into the DB.
        """
        path_obj = Path(file_path)
        filename = path_obj.name
        
        cursor = conn.cursor()
        
        # Insert file record
        try:
            cursor.execute("INSERT INTO files (filepath, filename) VALUES (?, ?)", (str(path_obj), filename))
            file_id = cursor.lastrowid
        except sqlite3.IntegrityError:
            print(f"File {filename} already ingested. Skipping.")
            return

        with open(file_path, 'r', encoding=self.encoding, errors='replace') as f:
            text = f.read()

        doc = self.nlp(text)
        
        target_pos = {'NOUN', 'ADJ', 'VERB'}
        
        # Prepare batch data
        
        for sent in doc.sents:
            sent_start = sent.start_char
            
            # Insert sentence with global offset
            cursor.execute(
                "INSERT INTO sentences (file_id, text, file_offset_start) VALUES (?, ?, ?)", 
                (file_id, sent.text, sent_start)
            )
            sentence_id = cursor.lastrowid
            
            sent_tokens_data = []
            for token in sent:
                if token.is_space:
                    continue
                    
                lemma = token.lemma_
                pos = token.pos_
                
                # Calculate relative offset
                relative_start = token.idx - sent_start
                
                sent_tokens_data.append((
                    sentence_id,
                    token.text,
                    lemma,
                    pos,
                    relative_start
                ))
            
            if sent_tokens_data:
                cursor.executemany(
                    "INSERT INTO tokens (sentence_id, text, lemma, pos, start_char) VALUES (?, ?, ?, ?, ?)",
                    sent_tokens_data
                )

    def preprocess_corpus(self, input_dir: str, db_path: str, max_files: int = None,
                          tqdm_class: Type[tqdm_std] = tqdm) -> Dict[str, Any]:
        """
        Processes text files in a directory and saves them to a SQLite DB.

        Args:
            input_dir: Path to directory containing text files.
            db_path: Output SQLite database path.
            max_files: Limit to N random files (for testing). None = all files.
            tqdm_class: Progress bar class to use (tqdm for CLI, stqdm for Streamlit).
        """
        input_path = Path(input_dir)
        output_db_path = Path(db_path)

        # Ensure parent dir exists
        output_db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._init_db(output_db_path)
        count = 0

        try:
            files = list(input_path.rglob('*.txt'))
            if max_files and max_files < len(files):
                import random
                files = random.sample(files, max_files)
                print(f"Randomly selected {max_files} files for ingestion.")

            total_files = len(files)
            print(f"Found {total_files} files to ingest.")

            with tqdm_class(total=total_files, desc="Ingesting", unit="file") as pbar:
                for txt_file in files:
                    # Wrap each file processing in a transaction
                    try:
                        self.process_file_to_db(str(txt_file), conn)
                        conn.commit()
                        count += 1
                    except Exception as e:
                        print(f"Error processing {txt_file}: {e}")
                        conn.rollback()

                    pbar.update(1)
                    pbar.set_postfix_str(txt_file.name[:30])

        finally:
            conn.close()

        print(f"Ingestion complete. Processed {count} files. Saved to {db_path}")
        return {"total_docs": count}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        ingestor = Ingestor()
        ingestor.preprocess_corpus(sys.argv[1], sys.argv[2])
