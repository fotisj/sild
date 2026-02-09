"""
Integration tests for the semantic change analysis pipeline.

These tests verify the end-to-end flow from data ingestion to analysis.
Note: Full integration tests with real models are slow and marked accordingly.
"""
import os
import sqlite3
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestCorpusToEmbeddingFlow:
    """Integration tests for corpus → embedding flow."""

    def test_corpus_query_returns_valid_samples(self):
        """Corpus query returns samples suitable for embedding extraction."""
        from semantic_change.corpus import Corpus
        import shutil

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = os.path.join(tmpdir, "test.db")

            # Create test database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("CREATE TABLE metadata (key TEXT, value TEXT)")
            cursor.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, filepath TEXT, filename TEXT)")
            cursor.execute("""
                CREATE TABLE sentences (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    text TEXT,
                    file_offset_start INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE tokens (
                    id INTEGER PRIMARY KEY,
                    sentence_id INTEGER,
                    text TEXT,
                    lemma TEXT,
                    pos TEXT,
                    start_char INTEGER
                )
            """)

            # Insert test data
            cursor.execute("INSERT INTO files VALUES (1, '/test.txt', 'test.txt')")
            cursor.execute(
                "INSERT INTO sentences VALUES (1, 1, 'The factory is large.', 0)"
            )
            cursor.execute(
                "INSERT INTO tokens VALUES (1, 1, 'factory', 'factory', 'NOUN', 4)"
            )

            conn.commit()
            conn.close()

            # Query samples
            corpus = Corpus("test", "", db_path)
            samples = corpus.query_samples("factory", n=10)

            # Verify sample format is compatible with embedder
            assert len(samples) == 1
            sample = samples[0]

            # Required fields for BertEmbedder
            assert "sentence" in sample
            assert "matched_word" in sample
            assert "start_char" in sample
            assert isinstance(sample["start_char"], int)
            assert sample["matched_word"] in sample["sentence"]

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestWsiClusteringFlow:
    """Integration tests for embedding → WSI clustering flow."""

    def test_embeddings_to_wsi_labels(self):
        """WSI correctly clusters embeddings into sense groups."""
        from semantic_change.wsi import WordSenseInductor

        # Create synthetic embeddings with 2 clear clusters
        np.random.seed(42)
        cluster1 = np.random.randn(15, 100) + np.array([5] * 100)
        cluster2 = np.random.randn(15, 100) + np.array([-5] * 100)
        embeddings = np.vstack([cluster1, cluster2]).astype(np.float32)

        # Run WSI
        wsi = WordSenseInductor(algorithm='kmeans', n_clusters=2)
        labels = wsi.fit_predict(embeddings)

        # Verify clustering
        assert len(labels) == 30
        assert len(set(labels)) == 2

        # First 15 should mostly be in one cluster
        # Last 15 should mostly be in another
        first_cluster = labels[:15]
        second_cluster = labels[15:]

        # Majority should be same within each half
        assert np.sum(first_cluster == first_cluster[0]) >= 12
        assert np.sum(second_cluster == second_cluster[0]) >= 12

    def test_substitute_wsi_clustering(self):
        """SubstituteWSI clusters substitutes into senses."""
        from semantic_change.wsi import SubstituteWSI

        # Substitutes with 2 clear semantic groups
        substitutes = [
            # Music sense
            ['guitar', 'bass', 'drum', 'instrument'],
            ['guitar', 'piano', 'music', 'instrument'],
            ['bass', 'drum', 'rhythm', 'music'],
            # Fish sense
            ['fish', 'salmon', 'trout', 'water'],
            ['bass', 'fish', 'perch', 'lake'],
            ['salmon', 'trout', 'fishing', 'river'],
        ]

        wsi = SubstituteWSI(min_community_size=2)
        labels = wsi.fit_predict(substitutes)

        assert len(labels) == 6
        # Should find at least some structure
        assert wsi.get_num_senses() >= 1


class TestStatsServiceIntegration:
    """Integration tests for stats service with real corpus."""

    def test_stats_service_reads_corpus(self):
        """StatsService correctly reads corpus statistics."""
        from semantic_change.services import StatsService

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "corpus.db")

            # Create minimal database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("CREATE TABLE files (id INTEGER PRIMARY KEY)")
            cursor.execute("CREATE TABLE sentences (id INTEGER PRIMARY KEY)")
            cursor.execute("CREATE TABLE tokens (id INTEGER PRIMARY KEY)")

            # Insert some records
            for i in range(5):
                cursor.execute("INSERT INTO files (id) VALUES (?)", (i,))
            for i in range(20):
                cursor.execute("INSERT INTO sentences (id) VALUES (?)", (i,))
            for i in range(100):
                cursor.execute("INSERT INTO tokens (id) VALUES (?)", (i,))

            conn.commit()
            conn.close()

            # Use StatsService
            service = StatsService()
            stats = service.get_corpus_stats(db_path, "test")

            assert stats is not None
            assert stats.files == 5
            assert stats.sentences == 20
            assert stats.tokens == 100


class TestConfigManagerIntegration:
    """Integration tests for config manager."""

    def test_config_roundtrip(self):
        """Config can be saved and loaded correctly."""
        from semantic_change.config_manager import AppConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")

            # Create config with custom values
            original = AppConfig(
                project_id="9999",
                model_name="custom-model",
                n_samples=150,
                wsi_algorithm="spectral"
            )
            original.save(config_path)

            # Load and verify
            loaded = AppConfig.load(config_path)

            assert loaded.project_id == "9999"
            assert loaded.model_name == "custom-model"
            assert loaded.n_samples == 150
            assert loaded.wsi_algorithm == "spectral"

    def test_config_db_path_integration(self):
        """Config correctly generates database paths."""
        from semantic_change.config_manager import AppConfig

        config = AppConfig(data_dir="my_data_dir")
        db_t1, db_t2 = config.get_db_paths()

        assert "my_data_dir" in db_t1
        assert "my_data_dir" in db_t2
        assert "corpus_t1.db" in db_t1
        assert "corpus_t2.db" in db_t2


class TestClusterServiceIntegration:
    """Integration tests for cluster service."""

    def test_save_and_load_cluster(self):
        """Cluster data can be saved and loaded correctly."""
        import json
        from semantic_change.services import ClusterService

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            embeddings = np.random.randn(10, 768).astype(np.float32)
            sentences = np.array([f"Sentence {i}" for i in range(10)])
            filenames = np.array([f"file{i}.txt" for i in range(10)])
            time_labels = np.array(["1800"] * 5 + ["1900"] * 5)
            sense_labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

            metadata = {
                "project_id": "1234",
                "model_name": "bert-base",
                "target_word": "test"
            }

            # Save cluster 0
            filepath = ClusterService.save_for_drilldown(
                embeddings=embeddings,
                sentences=sentences,
                filenames=filenames,
                time_labels=time_labels,
                sense_labels=sense_labels,
                spans=None,
                metadata=metadata,
                cluster_id=0,
                output_dir=tmpdir
            )

            # Load and verify
            with np.load(filepath, allow_pickle=True) as data:
                assert len(data["embeddings"]) == 5  # Only cluster 0
                assert len(data["sentences"]) == 5
                assert all(t == "1800" for t in data["time_labels"])

                saved_meta = json.loads(str(data["metadata"]))
                assert saved_meta["target_word"] == "test"
                assert saved_meta["original_cluster_id"] == 0


class TestDoubleBufferedWriteQueue:
    """Tests for the double-buffered write queue in process_corpus()."""

    def test_queue_based_processing_completes(self):
        """Double-buffered queue processes all chunks correctly."""
        from queue import Queue
        from threading import Thread
        import time

        # Simulate the queue-based pattern used in process_corpus()
        write_queue = Queue(maxsize=2)
        results_written = []
        write_error = [None]

        def writer_thread():
            try:
                while True:
                    item = write_queue.get()
                    if item is None:
                        break
                    results_written.append(item)
                    time.sleep(0.01)  # Simulate I/O
                    write_queue.task_done()
            except Exception as e:
                write_error[0] = e

        writer = Thread(target=writer_thread, daemon=True)
        writer.start()

        # Simulate processing 5 batches
        try:
            for i in range(5):
                batch_results = [f"embedding_{i}_{j}" for j in range(10)]
                write_queue.put(batch_results)

            write_queue.join()
            assert write_error[0] is None
        finally:
            write_queue.put(None)
            writer.join(timeout=5)

        # Verify all batches were written
        assert len(results_written) == 5
        assert all(len(batch) == 10 for batch in results_written)

    def test_queue_error_propagation(self):
        """Errors in writer thread are properly propagated."""
        from queue import Queue
        from threading import Thread

        write_queue = Queue(maxsize=2)
        write_error = [None]

        def failing_writer():
            try:
                item = write_queue.get()
                if item is not None:
                    raise ValueError("Simulated ChromaDB error")
            except Exception as e:
                write_error[0] = e

        writer = Thread(target=failing_writer, daemon=True)
        writer.start()

        # Submit work that will cause an error
        write_queue.put(["test_data"])
        writer.join(timeout=5)

        # Error should be captured
        assert write_error[0] is not None
        assert "Simulated ChromaDB error" in str(write_error[0])

    def test_queue_backpressure(self):
        """Queue with maxsize=2 provides backpressure when writes are slow."""
        from queue import Queue, Full
        from threading import Thread, Event

        # Use maxsize=1 for simpler test
        write_queue = Queue(maxsize=1)
        writer_blocked = Event()
        release_writer = Event()

        def blocking_writer():
            while True:
                item = write_queue.get()
                if item is None:
                    write_queue.task_done()
                    break
                writer_blocked.set()
                release_writer.wait(timeout=10)  # Block until released
                write_queue.task_done()

        writer = Thread(target=blocking_writer, daemon=True)
        writer.start()

        try:
            # Put first item - writer will pick it up and block
            write_queue.put("batch_1")
            writer_blocked.wait(timeout=5)

            # Now queue is empty but writer is blocked, so we can put one more
            write_queue.put("batch_2")

            # Queue is now full (maxsize=1), third put should fail
            with pytest.raises(Full):
                write_queue.put_nowait("batch_3")
        finally:
            release_writer.set()  # Allow writer to finish
            write_queue.put(None)  # Poison pill
            writer.join(timeout=5)

    def test_queue_cleanup_on_exception(self):
        """Writer thread exits cleanly when poison pill is sent after error."""
        from queue import Queue
        from threading import Thread

        write_queue = Queue(maxsize=2)
        writer_exited = [False]

        def writer_with_exit_flag():
            while True:
                item = write_queue.get()
                if item is None:
                    writer_exited[0] = True
                    break
                write_queue.task_done()

        writer = Thread(target=writer_with_exit_flag, daemon=True)
        writer.start()

        # Simulate normal operation then cleanup
        write_queue.put("batch_1")
        write_queue.join()

        # Send poison pill
        write_queue.put(None)
        writer.join(timeout=5)

        assert writer_exited[0] is True


class TestEndToEndMocked:
    """Mocked end-to-end tests for the full pipeline."""

    def test_ingestion_to_query_flow(self):
        """Test flow from ingestion to corpus query (mocked)."""
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "corpus.db"

            # Simulate what ingestor creates
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            cursor.execute("CREATE TABLE files (id INTEGER PRIMARY KEY, filepath TEXT, filename TEXT)")
            cursor.execute("""
                CREATE TABLE sentences (
                    id INTEGER PRIMARY KEY,
                    file_id INTEGER,
                    text TEXT,
                    file_offset_start INTEGER
                )
            """)
            cursor.execute("""
                CREATE TABLE tokens (
                    id INTEGER PRIMARY KEY,
                    sentence_id INTEGER,
                    text TEXT,
                    lemma TEXT,
                    pos TEXT,
                    start_char INTEGER
                )
            """)
            cursor.execute("CREATE INDEX idx_lemma ON tokens(lemma)")

            # Insert metadata
            cursor.execute(
                "INSERT INTO metadata VALUES ('spacy_model', 'en_core_web_sm')"
            )

            # Insert sample data simulating ingested text
            cursor.execute(
                "INSERT INTO files VALUES (1, '/book.txt', 'book.txt')"
            )
            cursor.execute(
                "INSERT INTO sentences VALUES (1, 1, 'The bank by the river is peaceful.', 0)"
            )
            cursor.execute(
                "INSERT INTO sentences VALUES (2, 1, 'The bank refused my loan application.', 40)"
            )
            cursor.execute(
                "INSERT INTO tokens VALUES (1, 1, 'bank', 'bank', 'NOUN', 4)"
            )
            cursor.execute(
                "INSERT INTO tokens VALUES (2, 2, 'bank', 'bank', 'NOUN', 4)"
            )

            conn.commit()
            conn.close()

            # Query samples using Corpus
            from semantic_change.corpus import Corpus
            corpus = Corpus("test", "", str(db_path))

            samples = corpus.query_samples("bank", n=10)

            # Verify we get both senses
            assert len(samples) == 2

            sentences = [s["sentence"] for s in samples]
            assert any("river" in s for s in sentences)
            assert any("loan" in s for s in sentences)

            if corpus.conn:
                corpus.conn.close()
                corpus.conn = None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_query_to_clustering_flow(self):
        """Test flow from corpus query to WSI clustering (mocked embeddings)."""
        from semantic_change.wsi import WordSenseInductor

        # Simulate samples from corpus
        samples = [
            {"sentence": "The bank by the river.", "matched_word": "bank", "start_char": 4},
            {"sentence": "A steep bank of earth.", "matched_word": "bank", "start_char": 8},
            {"sentence": "The bank approved the loan.", "matched_word": "bank", "start_char": 4},
            {"sentence": "My bank account is empty.", "matched_word": "bank", "start_char": 3},
        ]

        # Simulate embeddings (2 senses: river bank vs financial bank)
        np.random.seed(42)
        river_embeddings = np.random.randn(2, 768) + np.array([5] * 768)
        financial_embeddings = np.random.randn(2, 768) + np.array([-5] * 768)
        embeddings = np.vstack([river_embeddings, financial_embeddings]).astype(np.float32)

        # Run WSI
        wsi = WordSenseInductor(algorithm='kmeans', n_clusters=2)
        labels = wsi.fit_predict(embeddings)

        # Verify clustering separates the senses
        assert len(labels) == 4
        # First 2 (river bank) should be in same cluster
        assert labels[0] == labels[1]
        # Last 2 (financial bank) should be in same cluster
        assert labels[2] == labels[3]
        # The two groups should be in different clusters
        assert labels[0] != labels[2]
