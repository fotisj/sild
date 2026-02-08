"""
Tests for the corpus module.
"""
import os
import shutil
import sqlite3
import tempfile
import pytest

from semantic_change.corpus import (
    Corpus,
    CorpusManager,
    get_db_metadata,
    get_spacy_model_from_db,
)


def create_test_db(db_path: str) -> None:
    """Create a minimal test database with sample data."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create schema
    cursor.execute("""
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE,
            filename TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE sentences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
            text TEXT,
            file_offset_start INTEGER
        )
    """)
    cursor.execute("""
        CREATE TABLE tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sentence_id INTEGER,
            text TEXT,
            lemma TEXT,
            pos TEXT,
            start_char INTEGER
        )
    """)

    # Insert metadata
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("spacy_model", "en_core_web_sm"))
    cursor.execute("INSERT INTO metadata (key, value) VALUES (?, ?)",
                   ("encoding", "utf-8"))

    # Insert sample files
    cursor.execute("INSERT INTO files (filepath, filename) VALUES (?, ?)",
                   ("/path/to/file1.txt", "file1.txt"))
    cursor.execute("INSERT INTO files (filepath, filename) VALUES (?, ?)",
                   ("/path/to/file2.txt", "file2.txt"))

    # Insert sample sentences
    cursor.execute(
        "INSERT INTO sentences (file_id, text, file_offset_start) VALUES (?, ?, ?)",
        (1, "The factory produced many goods.", 0)
    )
    cursor.execute(
        "INSERT INTO sentences (file_id, text, file_offset_start) VALUES (?, ?, ?)",
        (1, "Workers entered the factory at dawn.", 35)
    )
    cursor.execute(
        "INSERT INTO sentences (file_id, text, file_offset_start) VALUES (?, ?, ?)",
        (2, "The old factory was demolished.", 0)
    )

    # Insert sample tokens (simplified - just key tokens)
    tokens = [
        # Sentence 1
        (1, "factory", "factory", "NOUN", 4),
        (1, "goods", "good", "NOUN", 27),
        # Sentence 2
        (2, "Workers", "worker", "NOUN", 0),
        (2, "factory", "factory", "NOUN", 20),
        # Sentence 3
        (3, "factory", "factory", "NOUN", 8),
    ]
    cursor.executemany(
        "INSERT INTO tokens (sentence_id, text, lemma, pos, start_char) VALUES (?, ?, ?, ?, ?)",
        tokens
    )

    conn.commit()
    conn.close()


class TestGetDbMetadata:
    """Tests for get_db_metadata function."""

    def test_returns_value_when_key_exists(self):
        """Returns the correct value for an existing key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            result = get_db_metadata(db_path, "spacy_model")
            assert result == "en_core_web_sm"

    def test_returns_none_for_missing_key(self):
        """Returns None for a key that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            result = get_db_metadata(db_path, "nonexistent_key")
            assert result is None

    def test_returns_none_for_missing_file(self):
        """Returns None when database file doesn't exist."""
        result = get_db_metadata("/nonexistent/path/db.db", "any_key")
        assert result is None


class TestGetSpacyModelFromDb:
    """Tests for get_spacy_model_from_db function."""

    def test_returns_spacy_model_name(self):
        """Returns the spaCy model name from metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            result = get_spacy_model_from_db(db_path)
            assert result == "en_core_web_sm"


class TestCorpus:
    """Tests for the Corpus class."""

    def test_init_connects_to_database(self):
        """Successfully connects to an existing database."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "/raw/path", db_path)
            assert corpus.conn is not None
            assert corpus.name == "test"

            # Close connection before cleanup
            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_init_handles_missing_database(self):
        """Handles missing database gracefully."""
        corpus = Corpus("test", "/raw/path", "/nonexistent/db.db")
        assert corpus.conn is None

    def test_get_stats_returns_counts(self):
        """Returns correct file, sentence, and token counts."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            stats = corpus.get_stats()

            assert stats["files"] == 2
            assert stats["sentences"] == 3
            assert stats["tokens"] == 5

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_stats_empty_without_connection(self):
        """Returns empty dict when no database connection."""
        corpus = Corpus("test", "", None)
        stats = corpus.get_stats()
        assert stats == {}

    def test_get_metadata(self):
        """Returns metadata values correctly."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            assert corpus.get_metadata("spacy_model") == "en_core_web_sm"
            assert corpus.get_metadata("encoding") == "utf-8"

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_spacy_model(self):
        """Returns the spaCy model name."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            assert corpus.get_spacy_model() == "en_core_web_sm"

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_top_lemmas(self):
        """Returns top lemmas by frequency for given POS."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            top_lemmas = corpus.get_top_lemmas(pos="NOUN", limit=10)

            # "factory" appears 3 times, should be first
            assert len(top_lemmas) > 0
            assert top_lemmas[0][0] == "factory"
            assert top_lemmas[0][1] == 3

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_query_samples_by_lemma(self):
        """Returns samples matching a lemma."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            samples = corpus.query_samples("factory", n=10)

            assert len(samples) == 3
            for sample in samples:
                assert "factory" in sample["sentence"].lower()
                assert sample["lemma"] == "factory"

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_query_samples_with_pos_filter(self):
        """Filters samples by POS tag."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            samples = corpus.query_samples("factory", n=10, pos_filter="NOUN")

            assert len(samples) == 3  # All factory tokens are NOUN

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_query_samples_exact_match(self):
        """Exact match searches by token text, not lemma."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)

            # Search for exact token "Workers" (capitalized)
            samples = corpus.query_samples("Workers", n=10, exact_match=True)
            assert len(samples) == 1
            assert samples[0]["matched_word"] == "Workers"

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_query_samples_returns_required_fields(self):
        """Returned samples have all required fields."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            samples = corpus.query_samples("factory", n=1)

            assert len(samples) == 1
            sample = samples[0]
            required_fields = ["sentence", "matched_word", "start_char",
                               "sentence_id", "file_offset_start", "file_path", "lemma"]
            for field in required_fields:
                assert field in sample

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_query_samples_empty_for_missing_word(self):
        """Returns empty list for word not in corpus."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            corpus = Corpus("test", "", db_path)
            samples = corpus.query_samples("nonexistent_word_xyz", n=10)
            assert samples == []

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestCorpusManager:
    """Tests for the CorpusManager class."""

    def test_add_and_get_corpus(self):
        """Can add and retrieve corpora."""
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")
            create_test_db(db_path)

            manager = CorpusManager()
            manager.add_corpus("1800", "/raw/1800", db_path)

            corpus = manager.get_corpus("1800")
            assert corpus is not None
            assert corpus.name == "1800"

            # Close connections
            for c in manager.corpora.values():
                if c.conn:
                    c.conn.close()
                    c.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_get_nonexistent_corpus(self):
        """Returns None for non-existent corpus."""
        manager = CorpusManager()
        assert manager.get_corpus("nonexistent") is None

    def test_multiple_corpora(self):
        """Can manage multiple corpora."""
        tmpdir = tempfile.mkdtemp()
        try:
            db1 = os.path.join(tmpdir, "db1.db")
            db2 = os.path.join(tmpdir, "db2.db")
            create_test_db(db1)
            create_test_db(db2)

            manager = CorpusManager()
            manager.add_corpus("1800", "/raw/1800", db1)
            manager.add_corpus("1900", "/raw/1900", db2)

            assert manager.get_corpus("1800") is not None
            assert manager.get_corpus("1900") is not None
            assert len(manager.corpora) == 2

            # Close connections
            for c in manager.corpora.values():
                if c.conn:
                    c.conn.close()
                    c.conn = None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
