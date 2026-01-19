import os
import random
import sqlite3
from typing import List, Dict, Optional, Any, Tuple


def get_db_metadata(db_path: str, key: str) -> Optional[str]:
    """Reads a metadata value directly from a database file."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def get_spacy_model_from_db(db_path: str) -> Optional[str]:
    """Returns the spaCy model name used during ingestion of a database."""
    return get_db_metadata(db_path, "spacy_model")

class Corpus:
    """Represents a single corpus (e.g., a specific time period) backed by a SQLite database."""
    
    def __init__(self, name: str, path: str, ingested_path: Optional[str] = None):
        self.name = name
        self.path = path # Path to raw files (kept for reference)
        self.ingested_path = ingested_path # Path to SQLite DB
        self.conn = None
        
        if ingested_path and os.path.exists(ingested_path):
            try:
                self.conn = sqlite3.connect(ingested_path, check_same_thread=False)
                # Enable WAL for read performance
                self.conn.execute("PRAGMA journal_mode=WAL;")
            except sqlite3.Error as e:
                print(f"Error connecting to database {ingested_path}: {e}")

    def __del__(self):
        if self.conn:
            self.conn.close()

    def get_stats(self) -> Dict[str, int]:
        """Returns basic statistics about the corpus from the DB."""
        if not self.conn:
            return {}

        cursor = self.conn.cursor()
        stats = {}
        try:
            cursor.execute("SELECT count(*) FROM files")
            stats['files'] = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM sentences")
            stats['sentences'] = cursor.fetchone()[0]
            cursor.execute("SELECT count(*) FROM tokens")
            stats['tokens'] = cursor.fetchone()[0]
        except sqlite3.Error:
            pass
        return stats

    def get_metadata(self, key: str) -> Optional[str]:
        """Returns a metadata value from the database."""
        if not self.conn:
            return None

        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT value FROM metadata WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
        except sqlite3.Error:
            return None

    def get_spacy_model(self) -> Optional[str]:
        """Returns the spaCy model name used during ingestion."""
        return self.get_metadata("spacy_model")

    def get_top_lemmas(self, pos: str = 'NOUN', limit: int = 2000) -> List[Tuple[str, int]]:
        """Returns the top frequent lemmas for a given POS tag."""
        if not self.conn:
            return []
            
        cursor = self.conn.cursor()
        query = """
            SELECT lemma, count(*) as freq 
            FROM tokens 
            WHERE pos = ? 
            GROUP BY lemma 
            ORDER BY freq DESC 
            LIMIT ?
        """
        cursor.execute(query, (pos, limit))
        return cursor.fetchall()

    def get_frequency_map(self) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, str], int]]:
        """
        Returns (term_freq, doc_freq) where keys are (lemma, pos).
        """
        if not self.conn:
            return {}, {}
            
        cursor = self.conn.cursor()
        target_pos = ('NOUN', 'VERB', 'ADJ')
        pos_placeholder = ','.join('?' for _ in target_pos)
        
        try:
            # Term Freq
            cursor.execute(f"SELECT lemma, pos, count(*) FROM tokens WHERE pos IN ({pos_placeholder}) GROUP BY lemma, pos", target_pos)
            term_freq = {(row[0], row[1]): row[2] for row in cursor.fetchall()}
            
            # Doc Freq
            cursor.execute(f"""
                SELECT t.lemma, t.pos, count(DISTINCT s.file_id)
                FROM tokens t
                JOIN sentences s ON t.sentence_id = s.id
                WHERE t.pos IN ({pos_placeholder})
                GROUP BY t.lemma, t.pos
            """, target_pos)
            doc_freq = {(row[0], row[1]): row[2] for row in cursor.fetchall()}
            
            return term_freq, doc_freq
        except sqlite3.Error as e:
            print(f"Error fetching frequency map: {e}")
            return {}, {}

    def query_samples(self, word: str, n: int = 50, pos_filter: str = None, exact_match: bool = False) -> List[Dict[str, str]]:
        """
        Retrieves n random samples containing the word.

        Args:
            word: The word to search for.
            n: Maximum number of samples to return.
            pos_filter: Optional POS tag filter (e.g., 'NOUN', 'VERB').
            exact_match: If True, search by exact token form (case-insensitive).
                        If False (default), search by lemma.

        Returns:
            List of dicts with keys: sentence, matched_word, start_char, sentence_id,
            file_offset_start, file_path, lemma.
        """
        if not self.conn:
            print("Error: No ingested database found. Please ingest the corpus first.")
            return []

        cursor = self.conn.cursor()

        # We need the sentence text, the specific token form, its start offset, and file info
        # Use exact token match or lemma match based on exact_match flag
        if exact_match:
            # Search by exact token form (case-sensitive)
            query = """
                SELECT s.text, t.text, t.start_char, s.id, s.file_offset_start, f.filepath, t.lemma
                FROM tokens t
                JOIN sentences s ON t.sentence_id = s.id
                JOIN files f ON s.file_id = f.id
                WHERE t.text = ?
            """
        else:
            # Search by lemma (case-sensitive to support cased models)
            query = """
                SELECT s.text, t.text, t.start_char, s.id, s.file_offset_start, f.filepath, t.lemma
                FROM tokens t
                JOIN sentences s ON t.sentence_id = s.id
                JOIN files f ON s.file_id = f.id
                WHERE t.lemma = ?
            """
        params = [word]
        
        if pos_filter:
            query += " AND t.pos = ?"
            params.append(pos_filter.upper())
            
        query += """
            ORDER BY RANDOM()
            LIMIT ?
        """
        params.append(n)
        
        try:
            cursor.execute(query, tuple(params))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "sentence": row[0],
                    "matched_word": row[1],
                    "start_char": row[2],
                    "sentence_id": row[3],
                    "file_offset_start": row[4],
                    "file_path": row[5],
                    "lemma": row[6]  # Use lemma from DB
                })
            return results
        except sqlite3.Error as e:
            print(f"Database error during query: {e}")
            return []

class CorpusManager:
    """Manages multiple corpora."""
    
    def __init__(self):
        self.corpora: Dict[str, Corpus] = {}

    def add_corpus(self, name: str, path: str, ingested_path: Optional[str] = None):
        self.corpora[name] = Corpus(name, path, ingested_path)

    def get_corpus(self, name: str) -> Optional[Corpus]:
        return self.corpora.get(name)