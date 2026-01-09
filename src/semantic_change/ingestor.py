import os
import sqlite3
import spacy
from typing import List, Dict, Any, Tuple
from pathlib import Path
from collections import Counter

class Ingestor:
    def __init__(self, model: str = "en_core_web_lg"):
        try:
            # Disable parser and ner for speed and memory efficiency
            print(f"Loading spaCy model: {model}")
            self.nlp = spacy.load(model, disable=["parser", "ner"])
            self.nlp.add_pipe('sentencizer')
            self.nlp.max_length = 6000000 # Increased limit
        except OSError:
            print(f"Model {model} not found. Please install it.")
            raise

    def _init_db(self, db_path: Path):
        """Initialize the SQLite database with the required schema."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable WAL mode for better concurrency/performance
        cursor.execute("PRAGMA journal_mode=WAL;")
        
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

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
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
                    
                lemma = token.lemma_.lower()
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

    def preprocess_corpus(self, input_dir: str, db_path: str, max_files: int = None) -> Dict[str, Any]:
        """
        Processes text files in a directory and saves them to a SQLite DB.
        """
        input_path = Path(input_dir)
        output_db_path = Path(db_path)
        
        # Ensure parent dir exists
        output_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = self._init_db(output_db_path)
        count = 0
        
        # Statistics (calculated on the fly or queryable later - let's do basic counting here)
        # Actually, for "all words" task, we can just query the DB later for stats.
        # But to match interface, we can return empty or basic stats.
        
        try:
            for txt_file in input_path.rglob('*.txt'):
                if max_files and count >= max_files:
                    break
                print(f"Processing {txt_file}...")
                
                # Wrap each file processing in a transaction
                try:
                    self.process_file_to_db(str(txt_file), conn)
                    conn.commit()
                    count += 1
                except Exception as e:
                    print(f"Error processing {txt_file}: {e}")
                    conn.rollback()
                    
        finally:
            conn.close()
        
        print(f"Ingestion complete. Processed {count} files. Saved to {db_path}")
        return {"total_docs": count}

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        ingestor = Ingestor()
        ingestor.preprocess_corpus(sys.argv[1], sys.argv[2])
