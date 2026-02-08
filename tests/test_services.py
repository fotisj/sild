"""
Tests for the services module.
"""
import json
import os
import shutil
import tempfile
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from semantic_change.services import (
    CorpusStats,
    EmbeddingStats,
    StatsService,
    ClusterService,
)


class TestCorpusStats:
    """Tests for the CorpusStats dataclass."""

    def test_creation(self):
        """Verify CorpusStats can be created with values."""
        stats = CorpusStats(files=100, sentences=5000, tokens=50000)

        assert stats.files == 100
        assert stats.sentences == 5000
        assert stats.tokens == 50000


class TestEmbeddingStats:
    """Tests for the EmbeddingStats dataclass."""

    def test_creation(self):
        """Verify EmbeddingStats can be created with values."""
        stats = EmbeddingStats(
            model_name="bert-base-uncased",
            total_embeddings=10000,
            unique_lemmas=500,
            count_t1=6000,
            count_t2=4000
        )

        assert stats.model_name == "bert-base-uncased"
        assert stats.total_embeddings == 10000
        assert stats.unique_lemmas == 500
        assert stats.count_t1 == 6000
        assert stats.count_t2 == 4000


class TestStatsService:
    """Tests for the StatsService class."""

    def test_init_default_path(self):
        """Verify default ChromaDB path is set."""
        service = StatsService()
        assert service.chroma_path == "data/chroma_db"

    def test_init_custom_path(self):
        """Verify custom ChromaDB path is accepted."""
        service = StatsService(chroma_path="custom/path")
        assert service.chroma_path == "custom/path"

    def test_get_corpus_stats_success(self):
        """Returns CorpusStats when corpus can be read."""
        with patch('semantic_change.corpus.Corpus') as MockCorpus:
            mock_corpus = MagicMock()
            mock_corpus.get_stats.return_value = {
                "files": 50,
                "sentences": 2000,
                "tokens": 20000
            }
            MockCorpus.return_value = mock_corpus

            service = StatsService()
            stats = service.get_corpus_stats("/path/to/db.db", "1800")

            assert stats is not None
            assert stats.files == 50
            assert stats.sentences == 2000
            assert stats.tokens == 20000

    def test_get_corpus_stats_failure(self):
        """Returns None when corpus cannot be read."""
        with patch('semantic_change.corpus.Corpus') as MockCorpus:
            MockCorpus.side_effect = Exception("Database error")

            service = StatsService()
            stats = service.get_corpus_stats("/path/to/bad.db", "1800")

            assert stats is None

    def test_get_embedding_stats_success(self):
        """Returns EmbeddingStats when embeddings can be read."""
        with patch('semantic_change.vector_store.VectorStore') as MockStore:
            mock_store = MagicMock()
            mock_store.count.side_effect = [100, 80]  # t1, t2

            # Mock collection for unique lemma counting
            mock_coll = MagicMock()
            mock_coll.get.return_value = {
                "metadatas": [
                    {"lemma": "word1"},
                    {"lemma": "word2"},
                    {"lemma": "word1"},  # duplicate
                ]
            }
            mock_store.get_or_create_collection.return_value = mock_coll

            MockStore.return_value = mock_store

            service = StatsService()
            stats = service.get_embedding_stats("1234", "bert-base-uncased")

            assert stats is not None
            assert stats.model_name == "bert-base-uncased"
            assert stats.total_embeddings == 180
            assert stats.count_t1 == 100
            assert stats.count_t2 == 80

    def test_get_embedding_stats_failure(self):
        """Returns None when embeddings cannot be read."""
        with patch('semantic_change.vector_store.VectorStore') as MockStore:
            MockStore.side_effect = Exception("ChromaDB error")

            service = StatsService()
            stats = service.get_embedding_stats("1234", "bert-base-uncased")

            assert stats is None


class TestClusterService:
    """Tests for the ClusterService class."""

    def test_save_for_drilldown_creates_file(self):
        """Verify save_for_drilldown creates correct .npz file."""
        tmpdir = tempfile.mkdtemp()
        try:
            # Create test data
            n_samples = 10
            embeddings = np.random.randn(n_samples, 768).astype(np.float32)
            sentences = np.array([f"Sentence {i}" for i in range(n_samples)])
            filenames = np.array([f"file{i}.txt" for i in range(n_samples)])
            time_labels = np.array(["1800"] * 5 + ["1900"] * 5)
            sense_labels = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
            spans = [(0, 5) for _ in range(n_samples)]

            metadata = {
                "project_id": "1234",
                "model_name": "bert-base-uncased",
                "target_word": "test"
            }

            # Save cluster 1
            filepath = ClusterService.save_for_drilldown(
                embeddings=embeddings,
                sentences=sentences,
                filenames=filenames,
                time_labels=time_labels,
                sense_labels=sense_labels,
                spans=spans,
                metadata=metadata,
                cluster_id=1,
                output_dir=tmpdir
            )

            # Verify file exists
            assert os.path.exists(filepath)
            assert filepath.endswith(".npz")

            # Verify file content - use context manager to ensure file is closed
            with np.load(filepath, allow_pickle=True) as data:
                assert "embeddings" in data
                assert "sentences" in data
                assert "metadata" in data

                # Should only contain cluster 1 items (4 items)
                assert len(data["embeddings"]) == 4
                assert len(data["sentences"]) == 4
        finally:
            # Manual cleanup - ignore errors on Windows
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_for_drilldown_filename_format(self):
        """Verify generated filename follows expected pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = np.random.randn(5, 768).astype(np.float32)
            sentences = np.array(["test"] * 5)
            filenames = np.array(["f.txt"] * 5)
            time_labels = np.array(["1800"] * 5)
            sense_labels = np.array([0, 0, 1, 1, 1])

            metadata = {
                "project_id": "5678",
                "model_name": "roberta-base",
                "target_word": "factory"
            }

            filepath = ClusterService.save_for_drilldown(
                embeddings=embeddings,
                sentences=sentences,
                filenames=filenames,
                time_labels=time_labels,
                sense_labels=sense_labels,
                spans=None,
                metadata=metadata,
                cluster_id=1,
                output_dir=tmpdir
            )

            filename = os.path.basename(filepath)
            assert "k5678" in filename
            assert "roberta_base" in filename
            assert "factory" in filename
            assert "cluster1" in filename

    def test_save_for_drilldown_without_spans(self):
        """Verify save works when spans is None."""
        tmpdir = tempfile.mkdtemp()
        try:
            embeddings = np.random.randn(3, 768).astype(np.float32)
            sentences = np.array(["a", "b", "c"])
            filenames = np.array(["1.txt", "2.txt", "3.txt"])
            time_labels = np.array(["1800", "1800", "1900"])
            sense_labels = np.array([0, 0, 0])

            metadata = {
                "project_id": "0001",
                "model_name": "test",
                "target_word": "word"
            }

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

            with np.load(filepath, allow_pickle=True) as data:
                assert data["spans"].item() is None
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_save_for_drilldown_metadata_includes_cluster_id(self):
        """Verify saved metadata includes original cluster ID."""
        tmpdir = tempfile.mkdtemp()
        try:
            embeddings = np.random.randn(2, 768).astype(np.float32)
            sentences = np.array(["a", "b"])
            filenames = np.array(["1.txt", "2.txt"])
            time_labels = np.array(["1800", "1900"])
            sense_labels = np.array([5, 5])

            metadata = {
                "project_id": "1111",
                "model_name": "m",
                "target_word": "w"
            }

            filepath = ClusterService.save_for_drilldown(
                embeddings=embeddings,
                sentences=sentences,
                filenames=filenames,
                time_labels=time_labels,
                sense_labels=sense_labels,
                spans=None,
                metadata=metadata,
                cluster_id=5,
                output_dir=tmpdir
            )

            with np.load(filepath, allow_pickle=True) as data:
                saved_metadata = json.loads(str(data["metadata"]))
                assert saved_metadata["original_cluster_id"] == 5
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
